import numpy as np
import math
import itertools
import matplotlib.pyplot as plt


# =========================
#   WEST FRONTIER (SW + NW)
#   Robust, KKT-consistent
# =========================

def _active_representation(mu, Sigma, active, tol=1e-12):
    """
    Given an active set of indices, compute:
      - parabola coefficients a,b,c: var(r) = a r^2 + b r + c
      - weights for active assets: w(r) = P * r + q
      - Markowitz scalars A,B,C,D and r_min
      - feasible r-interval [r_lo, r_hi] from w_i(r) >= 0
      - linear KKT multipliers for inactive assets: gamma_j(r) = g1 * r + g0
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)
    active = list(active)
    idx = np.array(active, dtype=int)

    cov = Sigma[np.ix_(idx, idx)]
    m = mu[idx]
    inv = np.linalg.inv(cov)
    e = np.ones(len(idx))

    # Markowitz scalars
    A = float(m @ inv @ m)
    B = float(m @ inv @ e)
    C = float(e @ inv @ e)
    D = A * C - B * B
    if D <= 0:
        raise ValueError("Degenerate active set (D <= 0).")

    # Parabola: var(r) = a r^2 + b r + c
    a = C / D
    b = -2.0 * B / D
    c = A / D

    # Active weights w(r) = P r + q
    v = inv @ m       # Σ^{-1} μ
    u = inv @ e       # Σ^{-1} 1
    P = (C / D) * v - (B / D) * u
    q = (-B / D) * v + (A / D) * u

    # Min-variance return for this active set
    r_min = B / C

    # Feasible r from weights >= 0 and asset-return bounds
    r_lo, r_hi = -1e18, 1e18
    for Pi, qi in zip(P, q):
        if abs(Pi) < 1e-14:
            if qi < -tol:
                raise ValueError("Active set gives negative weight for all r.")
            else:
                continue
        r0 = -qi / Pi
        if Pi > 0:
            # Pi r + qi >= 0 -> r >= r0
            r_lo = max(r_lo, r0)
        else:
            # Pi r + qi >= 0 -> r <= r0
            r_hi = min(r_hi, r0)

    r_lo = max(r_lo, float(m.min()))
    r_hi = min(r_hi, float(m.max()))
    if r_lo >= r_hi - tol:
        raise ValueError("No feasible r interval for this active set.")

    # KKT multipliers for inactive assets: gamma_j(r) = g1 * r + g0
    all_idx = np.arange(N)
    inactive = [j for j in all_idx if j not in active]
    gamma_coeffs = {}

    for j in inactive:
        s = Sigma[j, idx]  # Σ_{jS}
        mu_j = mu[j]
        # gamma_j(r) = s @ w_S(r) - α(r) * mu_j - β(r)
        # with α(r) = (C r - B) / D, β(r) = (A - B r) / D
        sp = float(s @ P)
        sq = float(s @ q)
        g1 = sp - (C / D) * mu_j + (B / D)
        g0 = sq + (B / D) * mu_j - A / D
        gamma_coeffs[j] = (g1, g0)

    return {
        "active": active,
        "inactive": inactive,
        "a": a, "b": b, "c": c,
        "P": P, "q": q,
        "A": A, "B": B, "C": C, "D": D,
        "r_min": r_min,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "gamma_coeffs": gamma_coeffs,
    }


def west_frontier_piecewise(mu, Sigma,
                            tol=1e-10,
                            verbose=False,
                            calc_ef=True,      # NEW: compute NW EF?
                            calc_low=True):    # NEW: compute SW frontier?
    """
    Robust, KKT-consistent west frontier (SW + NW) for long-only Markowitz.

    Returns:
      segments: list of dicts with keys:
        - 'parabola_idx' : integer ID
        - 'active_set'   : tuple of asset indices
        - 'lower_r', 'upper_r'
        - 'ef_frontier'  : 1 if NW efficient frontier, else 0
        - 'low_frontier' : 1 if SW frontier, else 0
        - 'ea_frontier'  : 0 here (east handled separately)
        - 'a_scaled', 'b_scaled', 'c_scaled', 'd_scaled' (=1.0)
      r_global: global min-variance return of the full N-asset set
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    segments = []
    parabola_idx_counter = 1

    def add_segment(active_set, a, b, c, r_low, r_high, ef, low, idx=None):
        nonlocal parabola_idx_counter, segments
        if r_high <= r_low + tol:
            return idx
        if idx is None:
            idx = parabola_idx_counter
            parabola_idx_counter += 1
        segments.append({
            "parabola_idx": idx,
            "active_set": tuple(active_set),
            "lower_r": float(r_low),
            "upper_r": float(r_high),
            "ef_frontier": int(ef),
            "low_frontier": int(low),
            "ea_frontier": 0,
            "a_scaled": float(a),
            "b_scaled": float(b),
            "c_scaled": float(c),
            "d_scaled": 1.0,
        })
        return idx

    # --- full active set + global min-variance r ---
    full_active = list(range(N))
    rep_full = _active_representation(mu, Sigma, full_active)
    r_global = rep_full["r_min"]
    if verbose:
        print(f"[WEST] N={N}, r_global={r_global}")

    # If neither NW nor SW requested, just return r_global and no segments
    if not calc_ef and not calc_low:
        return [], r_global

    # We'll reuse this index for the full-set parabola (so SW+NW pieces share it)
    full_idx = parabola_idx_counter
    parabola_idx_counter += 1

    # ========== NW (efficient frontier, r ≥ r_global) ==========
    if calc_ef:
        active = full_active.copy()
        rep = _active_representation(mu, Sigma, active)
        current_r = r_global

        while True:
            P, q = rep["P"], rep["q"]
            gamma = rep["gamma_coeffs"]
            r_hi = rep["r_hi"]

            candidates = []

            # 1) Exit events for active assets, moving r upward:
            for local_i, g_i in enumerate(rep["active"]):
                Pi, qi = P[local_i], q[local_i]
                if abs(Pi) < 1e-14:
                    continue
                if Pi < 0:
                    r0 = -qi / Pi
                    if r0 > current_r + tol and r0 <= r_hi + tol:
                        candidates.append((r0, ("exit", g_i)))

            # 2) Entry events for inactive assets, moving r upward:
            for j, (g1, g0) in gamma.items():
                if abs(g1) < 1e-14:
                    continue
                if g1 < 0:
                    r0 = -g0 / g1
                    if r0 > current_r + tol and r0 <= r_hi + tol:
                        candidates.append((r0, ("enter", j)))

            if not candidates:
                if len(active) == 1:
                    asset = active[0]
                    r_end = mu[asset]
                    a = b = 0.0
                    c = Sigma[asset, asset]
                else:
                    r_end = r_hi
                    a, b, c = rep["a"], rep["b"], rep["c"]
                add_segment(
                    active_set=active,
                    a=a, b=b, c=c,
                    r_low=current_r, r_high=r_end,
                    ef=True, low=False,
                    idx=full_idx if active == full_active else None,
                )
                break

            r_next, (etype, asset) = min(candidates, key=lambda x: x[0])

            if len(active) == 1:
                a = b = 0.0
                c = Sigma[active[0], active[0]]
            else:
                a, b, c = rep["a"], rep["b"], rep["c"]

            add_segment(
                active_set=active,
                a=a, b=b, c=c,
                r_low=current_r, r_high=r_next,
                ef=True, low=False,
                idx=full_idx if active == full_active else None,
            )

            if verbose:
                print(f"[WEST NW] r from {current_r} to {r_next}, event={etype} asset={asset}")

            # Update active set
            if etype == "exit":
                active = [i for i in active if i != asset]
            else:  # "enter"
                if asset not in active:
                    active = sorted(active + [asset])

            current_r = r_next

            if len(active) == 1:
                a = b = 0.0
                c = Sigma[active[0], active[0]]
                r_end = mu[active[0]]
                add_segment(
                    active_set=active,
                    a=a, b=b, c=c,
                    r_low=current_r, r_high=r_end,
                    ef=True, low=False,
                )
                break

            rep = _active_representation(mu, Sigma, active)

    # ========== SW (south-west frontier, r ≤ r_global) ==========
    if calc_low:
        active = full_active.copy()
        rep = _active_representation(mu, Sigma, active)
        current_r = r_global

        while True:
            P, q = rep["P"], rep["q"]
            gamma = rep["gamma_coeffs"]
            r_lo = rep["r_lo"]

            candidates = []

            # 1) Exit events for active assets, moving r downward:
            for local_i, g_i in enumerate(rep["active"]):
                Pi, qi = P[local_i], q[local_i]
                if abs(Pi) < 1e-14:
                    continue
                if Pi > 0:
                    r0 = -qi / Pi
                    if r0 < current_r - tol and r0 >= r_lo - tol:
                        candidates.append((r0, ("exit", g_i)))

            # 2) Entry events for inactive assets, moving r downward:
            for j, (g1, g0) in gamma.items():
                if abs(g1) < 1e-14:
                    continue
                if g1 > 0:
                    r0 = -g0 / g1
                    if r0 < current_r - tol and r0 >= r_lo - tol:
                        candidates.append((r0, ("enter", j)))

            if not candidates:
                if len(active) == 1:
                    asset = active[0]
                    r_end = mu[asset]
                    a = b = 0.0
                    c = Sigma[asset, asset]
                else:
                    r_end = r_lo
                    a, b, c = rep["a"], rep["b"], rep["c"]
                add_segment(
                    active_set=active,
                    a=a, b=b, c=c,
                    r_low=r_end, r_high=current_r,
                    ef=False, low=True,
                    idx=full_idx if active == full_active else None,
                )
                break

            r_next, (etype, asset) = max(candidates, key=lambda x: x[0])

            if len(active) == 1:
                a = b = 0.0
                c = Sigma[active[0], active[0]]
            else:
                a, b, c = rep["a"], rep["b"], rep["c"]

            add_segment(
                active_set=active,
                a=a, b=b, c=c,
                r_low=r_next, r_high=current_r,
                ef=False, low=True,
                idx=full_idx if active == full_active else None,
            )

            if verbose:
                print(f"[WEST SW] r from {r_next} to {current_r}, event={etype} asset={asset}")

            # Update active set
            if etype == "exit":
                active = [i for i in active if i != asset]
            else:  # "enter"
                if asset not in active:
                    active = sorted(active + [asset])

            current_r = r_next

            if len(active) == 1:
                a = b = 0.0
                c = Sigma[active[0], active[0]]
                r_end = mu[active[0]]
                add_segment(
                    active_set=active,
                    a=a, b=b, c=c,
                    r_low=r_end, r_high=current_r,
                    ef=False, low=True,
                )
                break

            rep = _active_representation(mu, Sigma, active)

    segments = sorted(segments, key=lambda s: (s["lower_r"], len(s["active_set"])))
    return segments, r_global

# =========================
#   EAST FRONTIER (max var)
#   Safe savings + grid
# =========================

def _prune_east_assets(mu, Sigma, tol=1e-12, verbose=False):
    """
    Safe pruning for east frontier:
      - group assets with identical (within tol) returns
      - for each group, keep only the one with largest diagonal variance
    This is safe because for east frontier, at that μ, the most volatile asset dominates.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    order = np.argsort(mu)
    keep = []
    i = 0
    while i < N:
        j = i + 1
        group = [order[i]]
        while j < N and abs(mu[order[j]] - mu[order[i]]) < tol:
            group.append(order[j])
            j += 1
        vars_diag = [Sigma[k, k] for k in group]
        best_idx = group[int(np.argmax(vars_diag))]
        keep.append(best_idx)
        i = j

    keep = sorted(keep)
    if verbose and len(keep) < N:
        print(f"[EAST] Pruned {N - len(keep)} duplicated-μ assets; kept {len(keep)}.")

    mu_e = mu[keep]
    Sigma_e = Sigma[np.ix_(keep, keep)]
    return keep, mu_e, Sigma_e


def _two_asset_parabola(mu, Sigma, i, j):
    """
    Exact 2-asset Markowitz parabola for assets (i,j).
    Returns (a,b,c) for var(r) = a r^2 + b r + c.
    """
    m = mu[[i, j]]
    cov = Sigma[np.ix_([i, j], [i, j])]
    inv = np.linalg.inv(cov)
    e = np.ones(2)

    A = float(m @ inv @ m)
    B = float(m @ inv @ e)
    C = float(e @ inv @ e)
    D = A * C - B * B
    if D <= 0:
        raise ValueError("Degenerate two-asset set.")

    a = C / D
    b = -2.0 * B / D
    c = A / D
    return a, b, c


def east_frontier_grid(mu, Sigma, K=200, tol=1e-10, verbose=False):
    """
    East frontier (max variance for each r) approximated on a grid of K returns
    between min(mu) and max(mu), using all 2-asset pairs of a pruned asset set.

    Safe savings:
      - Duplicated μ assets are pruned (keep max variance).

    Returns:
      segments: list of dicts with keys:
        - 'parabola_idx'
        - 'active_set'   : tuple (i,j) in ORIGINAL asset indices
        - 'lower_r', 'upper_r'
        - 'ef_frontier'  : 0
        - 'low_frontier' : 0
        - 'ea_frontier'  : 1
        - 'a_scaled', 'b_scaled', 'c_scaled', 'd_scaled' (=1.0)
      (approximate; accuracy set by K)
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    if N < 2:
        return [], None

    # Safe pruning by duplicated μ
    keep, mu_e, Sigma_e = _prune_east_assets(mu, Sigma, tol=tol, verbose=verbose)
    N_e = len(mu_e)
    if N_e < 2:
        return [], None

    r_min_all = float(mu_e.min())
    r_max_all = float(mu_e.max())
    grid = np.linspace(r_min_all, r_max_all, K)

    if verbose:
        print(f"[EAST-GRID] N={N}, N_pruned={N_e}, K={K}, r in [{r_min_all}, {r_max_all}]")

    # Precompute pairs in pruned index space and map back to original indices
    pairs = []
    for i_e, j_e in itertools.combinations(range(N_e), 2):
        i_orig = keep[i_e]
        j_orig = keep[j_e]
        pairs.append((i_e, j_e, i_orig, j_orig))

    best_pair_idx = []
    best_var = []

    for r in grid:
        max_var = -float("inf")
        best_idx = None

        for idx_pair, (i_e, j_e, i_orig, j_orig) in enumerate(pairs):
            mu_i = mu_e[i_e]
            mu_j = mu_e[j_e]
            m_lo = min(mu_i, mu_j)
            m_hi = max(mu_i, mu_j)
            if r < m_lo - tol or r > m_hi + tol:
                continue
            if abs(mu_i - mu_j) < 1e-14:
                continue  # shouldn't happen after pruning, but safe

            # weights to hit target r
            w_i = (r - mu_j) / (mu_i - mu_j)
            w_j = 1.0 - w_i
            if w_i < -tol or w_j < -tol:
                continue

            var = (
                w_i**2 * Sigma[i_orig, i_orig] +
                w_j**2 * Sigma[j_orig, j_orig] +
                2.0 * w_i * w_j * Sigma[i_orig, j_orig]
            )
            if var > max_var + 1e-14:
                max_var = var
                best_idx = idx_pair

        best_pair_idx.append(best_idx)
        best_var.append(max_var)

    segments = []
    parabola_idx_counter = 1
    k0 = 0
    while k0 < len(grid):
        pair_idx = best_pair_idx[k0]
        if pair_idx is None:
            k0 += 1
            continue

        k1 = k0 + 1
        while k1 < len(grid) and best_pair_idx[k1] == pair_idx:
            k1 += 1

        r_low = grid[k0]
        r_high = grid[k1 - 1]
        if r_high <= r_low + tol:
            k0 = k1
            continue

        i_e, j_e, i_orig, j_orig = pairs[pair_idx]
        a, b, c = _two_asset_parabola(mu, Sigma, i_orig, j_orig)

        segments.append({
            "parabola_idx": parabola_idx_counter,
            "active_set": (i_orig, j_orig),
            "lower_r": float(r_low),
            "upper_r": float(r_high),
            "ef_frontier": 0,
            "low_frontier": 0,
            "ea_frontier": 1,
            "a_scaled": float(a),
            "b_scaled": float(b),
            "c_scaled": float(c),
            "d_scaled": 1.0,
        })

        parabola_idx_counter += 1
        k0 = k1

    return segments, (r_min_all, r_max_all)

def east_frontier_exact(mu, Sigma, tol=1e-10, verbose=False):
    """
    Exact east frontier as the upper envelope of all 2-asset parabolas
    (after safe μ-duplication pruning). This is conceptually exact but
    scales poorly (~O(N^4)); only use for small N.

    Returns:
      segments: list of dicts with ea_frontier=1, analogous format to west.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    if N < 2:
        return [], None

    keep, mu_e, Sigma_e = _prune_east_assets(mu, Sigma, tol=tol, verbose=verbose)
    N_e = len(mu_e)
    if N_e < 2:
        return [], None

    r_min_all = float(mu_e.min())
    r_max_all = float(mu_e.max())

    # Build all two-asset parabolas in the pruned space, map back to original indices
    parabs = []
    idx_counter = 1
    for i_e, j_e in itertools.combinations(range(N_e), 2):
        i_orig = keep[i_e]
        j_orig = keep[j_e]
        m = mu[[i_orig, j_orig]]
        cov = Sigma[np.ix_([i_orig, j_orig], [i_orig, j_orig])]
        inv = np.linalg.inv(cov)
        e = np.ones(2)

        A = float(m @ inv @ m)
        B = float(m @ inv @ e)
        C = float(e @ inv @ e)
        D = A * C - B * B
        if D <= 0:
            continue

        a = C / D
        b = -2.0 * B / D
        c = A / D

        r_lo = float(min(m))
        r_hi = float(max(m))
        if r_lo >= r_hi - tol:
            continue

        parabs.append({
            "idx": idx_counter,
            "pair": (i_orig, j_orig),
            "a": a, "b": b, "c": c,
            "r_lo": r_lo,
            "r_hi": r_hi,
        })
        idx_counter += 1

    if not parabs:
        return [], None

    # Helper
    def _var_on(p, r):
        return p["a"] * r * r + p["b"] * r + p["c"]

    # Collect breakpoints: endpoints + pairwise intersections
    breaks = set([r_min_all, r_max_all])
    for p in parabs:
        breaks.add(p["r_lo"])
        breaks.add(p["r_hi"])

    for i, p in enumerate(parabs):
        for j in range(i + 1, len(parabs)):
            q = parabs[j]
            A = p["a"] - q["a"]
            B = p["b"] - q["b"]
            Cc = p["c"] - q["c"]
            if abs(A) < 1e-14 and abs(B) < 1e-14:
                continue
            roots = []
            if abs(A) < 1e-14:
                roots = [-Cc / B]
            else:
                disc = B * B - 4.0 * A * Cc
                if disc < 0:
                    continue
                sdisc = math.sqrt(max(0.0, disc))
                roots = [(-B - sdisc) / (2 * A), (-B + sdisc) / (2 * A)]
            for r in roots:
                if r_min_all - tol <= r <= r_max_all + tol:
                    breaks.add(r)

    breaks = sorted(breaks)
    merged = []
    for r in breaks:
        if not merged or abs(r - merged[-1]) > 1e-8:
            merged.append(r)
    breaks = merged

    segments = []
    parabola_idx_counter = 1
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid = 0.5 * (lo + hi)
        # feasible at mid?
        feas = [
            p for p in parabs
            if p["r_lo"] - tol <= mid <= p["r_hi"] + tol
        ]
        if not feas:
            continue
        best = max(feas, key=lambda p: _var_on(p, mid))

        segments.append({
            "parabola_idx": parabola_idx_counter,
            "active_set": best["pair"],
            "lower_r": float(lo),
            "upper_r": float(hi),
            "ef_frontier": 0,
            "low_frontier": 0,
            "ea_frontier": 1,
            "a_scaled": float(best["a"]),
            "b_scaled": float(best["b"]),
            "c_scaled": float(best["c"]),
            "d_scaled": 1.0,
        })
        parabola_idx_counter += 1

    if verbose:
        print(f"[EAST-EXACT] built {len(segments)} east segments from {len(parabs)} pairs.")

    return segments, (r_min_all, r_max_all)


# =========================
#   MASTER FUNCTION
# =========================


def compute_frontiers(mu, Sigma,
                      east_mode="grid",
                      east_K=200,
                      ef_frontier=True,
                      low_frontier=True,
                      weights=None,
                      sd=True,
                      w_deltas=False,
                      verbose=False,
                      graph=False):   
    """
    Compute west (SW + NW) and optionally east (max-var) frontiers.

    Returns
    -------
    {
      "segments": [ ... segment dicts ... ],
      "r_global": r_global,
      "w_diagnostics": { ... } or None
    }
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)

    # 1) WEST FRONTIER
    if ef_frontier or low_frontier:
        west_segments, r_global = west_frontier_piecewise(
            mu, Sigma,
            verbose=verbose,
            calc_ef=ef_frontier,
            calc_low=low_frontier
        )
    else:
        full_active = list(range(len(mu)))
        rep_full = _active_representation(mu, Sigma, full_active)
        r_global = rep_full["r_min"]
        west_segments = []
        if verbose:
            print("[MASTER] Skipping west frontier; only assets/east may be graphed.")

    # 2) EAST FRONTIER
    east_segments = []
    if east_mode == "grid":
        if isinstance(east_K, int) and east_K > 1:
            if verbose:
                print(f"[MASTER] Computing grid-based east frontier with K={east_K}.")
            east_segments, _ = east_frontier_grid(mu, Sigma, K=east_K, verbose=verbose)
        else:
            if verbose:
                print("[MASTER] east_mode='grid' but east_K not a valid int > 1; skipping east.")
    elif east_mode is True:
        if verbose:
            print("[MASTER] Computing exact east frontier (pairwise envelope).")
        east_segments, _ = east_frontier_exact(mu, Sigma, verbose=verbose)
    else:
        if verbose:
            print("[MASTER] Skipping east frontier (east_mode is False/None).")

    segments = west_segments + east_segments

    # 3) Portfolio diagnostics, if weights supplied
    w_diagnostics = None
    if weights is not None:
        w_diagnostics = _analyze_portfolio_weights(
            weights, mu, Sigma, segments, sd=sd, verbose=verbose
        )

        r_w = w_diagnostics["r_w"]
        sd_w = w_diagnostics["sd_w"]

        r_w = w_diagnostics["r_w"]
        sd_w = w_diagnostics["sd_w"]

        print("\n=== Portfolio Diagnostics ===")
        print(f"Portfolio r_w = {r_w}")
        print(f"Portfolio sd_w = {sd_w}")

        # If requested, show the portfolio weights themselves
        if w_deltas:
            base_w = np.asarray(weights, float)
            print("Portfolio w  =", base_w.tolist())

        print("")

        ef_same_var = w_diagnostics["ef_same_var"]
        fsr = w_diagnostics["frontier_same_r"]
        nearest = w_diagnostics["nearest_ef_point"]
        closest = w_diagnostics["closest_in_weights"]

        # 1) Frontier at same variance (same sd as w)
        print("1) Frontier at same variance (same sd as w):")
        if ef_same_var["exists"]:
            src = "EF" if ef_same_var.get("on_ef") else ("East" if ef_same_var.get("on_ea") else "unknown")
            print(f"   frontier source = {src}")
            print(f"   r_frontier      = {ef_same_var['r_frontier']}")
            print(f"   r_diff          = {ef_same_var['r_diff']}")
            print(f"   dissimilarity D = {ef_same_var['dissimilarity']}")
        else:
            print(f"   [no match] reason: {ef_same_var['reason']}")
        print("")

        # 2) Frontier at same return r_w
        print("2) Frontier at same return r_w:")
        if fsr["exists"]:
            src = "EF" if fsr["on_ef"] else ("SW" if fsr["on_low"] else "unknown")
            print(f"   frontier source = {src}")
            print(f"   sd_frontier     = {fsr['sd_frontier']}")
            print(f"   sd_diff         = {fsr['sd_diff']}")
            print(f"   dissimilarity D = {fsr['dissimilarity']}")
        else:
            print(f"   [no match] reason: {fsr['reason']}")
        print("")

        # 3 & 4 conceptually depend on the efficient frontier.
        # If ef_frontier=False, we still run _analyze_portfolio_weights,
        # but for UX we replace detailed 3 & 4 output with a clear message.
        if not ef_frontier:
            # 3) Nearest EF point in (r, sd) space
            print("3) Nearest EF point in (r, sd) space:")
            print("   Efficient frontier not calculated.")
            print("")

            # 4) Closest EF point in WEIGHT space
            print("4) Closest EF point in WEIGHT space (min D):")
            print("   Efficient frontier not calculated.")
            print("")
        else:
            # 3) Nearest EF point in (r, sd) space
            print("3) Nearest EF point in (r, sd) space:")
            if nearest["exists"]:
                print(f"   r_ef            = {nearest['r_ef']}")
                print(f"   sd_ef           = {nearest['sd_ef']}")
                print(f"   r_diff          = {nearest['r_diff']}")
                print(f"   sd_diff         = {nearest['sd_diff']}")
                print(f"   distance        = {nearest['distance']}")
                print(f"   dissimilarity D = {nearest['dissimilarity']}")
            else:
                print(f"   [no EF] reason: {nearest['reason']}")
            print("")

            # 4) Closest EF point in WEIGHT space
            print("4) Closest EF point in WEIGHT space (min D):")
            if closest["exists"]:
                print(f"   r_ef            = {closest['r_ef']}")
                print(f"   sd_ef           = {closest['sd_ef']}")
                print(f"   r_diff          = {closest['r_diff']}")
                print(f"   sd_diff         = {closest['sd_diff']}")
                print(f"   dissimilarity D = {closest['dissimilarity']}")
            else:
                print(f"   [no EF] reason: {closest['reason']}")
            print("")

        # Optional: weight vectors and deltas (only when w_deltas=True)
        if w_deltas:
            base_w = np.asarray(weights, float)

            print("=== Weight Details (w_deltas=True) ===")

            if ef_same_var.get("w_frontier") is not None:
                w1 = np.asarray(ef_same_var["w_frontier"], float)
                print("\n[1] w_f & Δw to same-variance frontier:")
                print("   w_f:", w1.tolist())
                print("   Δw: ", (w1 - base_w).tolist())

            if fsr.get("w_frontier") is not None:
                w2 = np.asarray(fsr["w_frontier"], float)
                print("\n[2] w_f & Δw to same-return frontier:")
                print("   w_f:", w2.tolist())
                print("   Δw: ", (w2 - base_w).tolist())

            # For 3 and 4, only show weight deltas when EF was actually computed.
            if ef_frontier:
                if nearest.get("w_ef") is not None:
                    w3 = np.asarray(nearest["w_ef"], float)
                    print("\n[3] w_f & Δw to nearest EF (r,sd) portfolio:")
                    print("   w_f:", w3.tolist())
                    print("   Δw: ", (w3 - base_w).tolist())

                if closest.get("w_ef") is not None:
                    w4 = np.asarray(closest["w_ef"], float)
                    print("\n[4] w_f & Δw to closest-in-weights EF portfolio:")
                    print("   w_f:", w4.tolist())
                    print("   Δw: ", (w4 - base_w).tolist())

        print("============================\n")

    # 4) Optional graph (ALWAYS after printing)
    if graph:
        _plot_frontiers(mu, Sigma, segments, r_global,
                        sd=sd,
                        weights=weights,
                        w_diagnostics=w_diagnostics)

    return {
        "segments": segments,
        "r_global": r_global,
        "w_diagnostics": w_diagnostics,
    }

# =========================
#   W* Portfolio Diagnostics
# =========================

def _analyze_portfolio_weights(weights, mu, Sigma, segments,
                               sd=True, verbose=False, tol=1e-10):
    """
    Analyze a given portfolio w relative to the frontiers.

    If sd=True, diagnostics are reported in terms of standard deviation:
      - r_w, sd_w
      - Frontier points reported as (r, sd) where applicable
      - 'nearest_ef_point' distance measured in (r, sd) space.

    Dissimilarity index:
      D = 0.5 * sum_i |w_i - w_i^*|
      for each diagnostic's corresponding frontier portfolio w^*.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    w = np.asarray(weights, float)
    if w.ndim > 1:
        if w.shape == (N, 1) or w.shape == (1, N):
            w = w.ravel()
        else:
            raise ValueError(f"weights has incompatible shape {w.shape}; expected length {N}.")
    if w.shape[0] != N:
        raise ValueError(f"weights length {w.shape[0]} != N={N}.")

    # basic portfolio stats
    r_w = float(mu @ w)
    var_w = float(w @ (Sigma @ w))
    sd_w = math.sqrt(max(var_w, 0.0)) if sd else None

    # r* (global min‐variance return) from full active set
    rep_full = _active_representation(mu, Sigma, list(range(N)))
    r_global = rep_full["r_min"]

    # variance of max‐return asset (global)
    asset_vars = np.diag(Sigma)
    idx_max_r = int(np.argmax(mu))
    var_max_r = float(asset_vars[idx_max_r])

    if verbose:
        if sd:
            print(f"[W-PORT] r_w={r_w}, sd_w={sd_w}, var_w={var_w}, sum(w)={w.sum()}, r_global={r_global}")
        else:
            print(f"[W-PORT] r_w={r_w}, var_w={var_w}, sum(w)={w.sum()}, r_global={r_global}")

    def var_on_seg(seg, r):
        a = seg["a_scaled"]
        b = seg["b_scaled"]
        c = seg["c_scaled"]
        return a * r * r + b * r + c

    # cache representations per active set to avoid recomputation
    rep_cache = {}

    def get_rep(active_set):
        key = tuple(active_set)
        if key not in rep_cache:
            rep_cache[key] = _active_representation(mu, Sigma, key)
        return rep_cache[key]

    def weights_on_segment(seg, r_star):
        """Return full N-vector of weights for a given segment at return r_star."""
        active = seg["active_set"]
        rep = get_rep(active)
        P, q = rep["P"], rep["q"]
        w_star = np.zeros(N)
        w_star[list(active)] = P * r_star + q
        return w_star

    def dissimilarity(w_star):
        """D = 0.5 * sum |w_i - w*_i|."""
        return 0.5 * float(np.sum(np.abs(w - w_star)))

    ef_segments  = [s for s in segments if s.get("ef_frontier", 0) == 1]
    low_segments = [s for s in segments if s.get("low_frontier", 0) == 1]
    ea_segments  = [s for s in segments if s.get("ea_frontier", 0) == 1]

    has_ef  = bool(ef_segments)
    has_low = bool(low_segments)
    has_ea  = bool(ea_segments)

    # -------------------------
    # 1) Frontier r at same variance (EF if var_w ≤ var_max_r, else East)
    # -------------------------
    ef_same_var = {
        "exists": False,
        "on_ef": False,
        "on_ea": False,
        "r_frontier": None,
        "r_diff": None,
        "dissimilarity": None,
        "w_frontier": None,
        "reason": None,
    }

    # Decide relevant frontier by variance
    use_ef_for_var = (var_w <= var_max_r)
    if use_ef_for_var:
        segments_for_var = ef_segments
    else:
        segments_for_var = ea_segments

    if not segments_for_var:
        # Relevant frontier missing entirely
        if use_ef_for_var:
            ef_same_var["reason"] = "Efficient frontier not calculated."
        else:
            ef_same_var["reason"] = "East frontier not calculated."
    else:
        roots = []
        seg_roots = []  # (seg, r0)
        for seg in segments_for_var:
            a = seg["a_scaled"]
            b = seg["b_scaled"]
            c = seg["c_scaled"]
            A = a
            B = b
            Cc = c - var_w

            if abs(A) < 1e-14:
                if abs(B) < 1e-14:
                    continue
                r0 = -Cc / B
                if seg["lower_r"] - tol <= r0 <= seg["upper_r"] + tol:
                    roots.append(r0)
                    seg_roots.append((seg, r0))
            else:
                disc = B * B - 4.0 * A * Cc
                if disc < 0:
                    continue
                sdisc = math.sqrt(max(0.0, disc))
                r1 = (-B - sdisc) / (2.0 * A)
                r2 = (-B + sdisc) / (2.0 * A)
                for r0 in (r1, r2):
                    if seg["lower_r"] - tol <= r0 <= seg["upper_r"] + tol:
                        roots.append(r0)
                        seg_roots.append((seg, r0))

        if seg_roots:
            # choose the root whose variance is closest to var_w (should be equal)
            best_seg = None
            best_r = None
            best_err = float("inf")
            best_v = None
            for seg, r0 in seg_roots:
                v0 = var_on_seg(seg, r0)
                err = abs(v0 - var_w)
                if err < best_err:
                    best_err = err
                    best_r = r0
                    best_v = v0
                    best_seg = seg
            if best_seg is not None:
                w_front = weights_on_segment(best_seg, best_r)
                D1 = dissimilarity(w_front)
                ef_same_var["exists"] = True
                ef_same_var["r_frontier"] = float(best_r)
                ef_same_var["r_diff"] = float(best_r - r_w)
                ef_same_var["dissimilarity"] = float(D1)
                ef_same_var["w_frontier"] = w_front.tolist()
                if best_seg.get("ef_frontier", 0) == 1:
                    ef_same_var["on_ef"] = True
                if best_seg.get("ea_frontier", 0) == 1:
                    ef_same_var["on_ea"] = True
        else:
            ef_same_var["reason"] = "Portfolio variance lies outside frontier variance range (or no intersection)."

    # -----------------------------------------------
    # 2) Frontier variance at same return r_w
    #    Relevant frontier by r_w vs r_global:
    #      - r_w < r*  -> SW
    #      - r_w >= r* -> EF
    # -----------------------------------------------
    frontier_same_r = {
        "exists": False,
        "on_ef": False,
        "on_low": False,
        "sd_frontier": None,
        "sd_diff": None,
        "dissimilarity": None,
        "w_frontier": None,
        "reason": None,
    }

    chosen_seg = None

    if r_w < r_global:
        # Relevant: SW frontier
        if not has_low:
            frontier_same_r["reason"] = "Low frontier not calculated."
        else:
            candidates = [s for s in low_segments
                          if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
            if candidates:
                chosen_seg = min(candidates, key=lambda s: var_on_seg(s, r_w))
                frontier_same_r["on_low"] = True
            else:
                frontier_same_r["reason"] = "Return r_w lies outside SW frontier r-range."
    else:
        # Relevant: EF frontier
        if not has_ef:
            frontier_same_r["reason"] = "Efficient frontier not calculated."
        else:
            candidates = [s for s in ef_segments
                          if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
            if candidates:
                chosen_seg = min(candidates, key=lambda s: var_on_seg(s, r_w))
                frontier_same_r["on_ef"] = True
            else:
                frontier_same_r["reason"] = "Return r_w lies outside EF frontier r-range."

    if chosen_seg is not None:
        v_front = var_on_seg(chosen_seg, r_w)
        sd_front = math.sqrt(max(v_front, 0.0)) if sd else None
        w_front = weights_on_segment(chosen_seg, r_w)
        D2 = dissimilarity(w_front)
        frontier_same_r["exists"] = True
        frontier_same_r["sd_frontier"] = sd_front
        frontier_same_r["sd_diff"] = None if not sd else (sd_w - sd_front)
        frontier_same_r["dissimilarity"] = float(D2)
        frontier_same_r["w_frontier"] = w_front.tolist()

    # ----------------------------------------------------
    # 3) Nearest EF point (continuous min over EF segments)
    # ----------------------------------------------------
    nearest_ef_point = {
        "exists": False,
        "r_ef": None,
        "sd_ef": None,
        "r_diff": None,
        "sd_diff": None,
        "distance": None,        # distance in chosen geometry
        "dissimilarity": None,
        "w_ef": None,
        "reason": None,
    }

    if not ef_segments:
        nearest_ef_point["reason"] = "No ef_frontier segments to project onto."
    else:
        # distance^2 on a given segment at return r
        def dist2_on_seg(seg, r):
            v = var_on_seg(seg, r)
            if not sd:
                # geometry in (r, var)
                dr = r - r_w
                dv = v - var_w
                return dr * dr + dv * dv
            else:
                # geometry in (r, sd)
                s = math.sqrt(max(v, 0.0))
                dr = r - r_w
                ds = s - sd_w
                return dr * dr + ds * ds

        # 1-D golden-section minimization on [r_lo, r_hi]
        def minimize_on_segment(seg, r_lo, r_hi, max_iter=60):
            if r_hi <= r_lo + tol:
                return None, None

            phi = (1.0 + math.sqrt(5.0)) / 2.0
            inv_phi = 1.0 / phi

            a = r_lo
            b = r_hi
            c = b - (b - a) * inv_phi
            d = a + (b - a) * inv_phi

            f_c = dist2_on_seg(seg, c)
            f_d = dist2_on_seg(seg, d)

            for _ in range(max_iter):
                if abs(b - a) < 1e-12:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - (b - a) * inv_phi
                    f_c = dist2_on_seg(seg, c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + (b - a) * inv_phi
                    f_d = dist2_on_seg(seg, d)

            r_star = 0.5 * (a + b)
            f_star = dist2_on_seg(seg, r_star)
            return r_star, f_star

        best_dist2 = float("inf")
        best_r = None
        best_v = None
        best_seg = None

        # iterate over all EF segments
        for seg in ef_segments:
            r_lo = seg["lower_r"]
            r_hi = seg["upper_r"]
            if r_hi <= r_lo + tol:
                continue

            r_star, f_star = minimize_on_segment(seg, r_lo, r_hi)
            if r_star is None:
                continue

            if f_star < best_dist2:
                best_dist2 = f_star
                best_r = float(r_star)
                best_v = float(var_on_seg(seg, r_star))
                best_seg = seg

        if best_seg is not None:
            sd_ef = math.sqrt(max(best_v, 0.0)) if sd else None
            w_ef = weights_on_segment(best_seg, best_r)
            D3 = dissimilarity(w_ef)

            nearest_ef_point["exists"] = True
            nearest_ef_point["r_ef"] = best_r
            nearest_ef_point["sd_ef"] = sd_ef
            nearest_ef_point["r_diff"] = best_r - r_w
            nearest_ef_point["sd_diff"] = None if not sd else (sd_ef - sd_w)
            nearest_ef_point["distance"] = math.sqrt(best_dist2)
            nearest_ef_point["dissimilarity"] = float(D3)
            nearest_ef_point["w_ef"] = w_ef.tolist()
        else:
            nearest_ef_point["reason"] = "No valid candidate points found on EF segments."

    # ----------------------------------------------------
    # 4) Closest EF point in WEIGHT space (min dissimilarity)
    # ----------------------------------------------------
    closest_in_weights = {
        "exists": False,
        "r_ef": None,
        "var_ef": None,
        "sd_ef": None,
        "r_diff": None,
        "var_diff": None,
        "sd_diff": None,
        "dissimilarity": None,
        "w_ef": None,
        "w_diff": None,
        "reason": None,
    }

    if not ef_segments:
        closest_in_weights["reason"] = "No ef_frontier segments to search in weight space."
    else:
        best_D = float("inf")
        best_r = None
        best_v = None
        best_w = None

        for seg in ef_segments:
            r_lo = seg["lower_r"]
            r_hi = seg["upper_r"]
            if r_hi <= r_lo + tol:
                continue
            rs = np.linspace(r_lo, r_hi, 100)
            for r0 in rs:
                w_star = weights_on_segment(seg, r0)
                D = dissimilarity(w_star)
                if D < best_D:
                    best_D = D
                    best_r = float(r0)
                    best_v = float(var_on_seg(seg, r0))
                    best_w = w_star.copy()

        if best_w is not None:
            sd_ef = math.sqrt(max(best_v, 0.0)) if sd else None
            w_diff = best_w - w
            closest_in_weights["exists"] = True
            closest_in_weights["r_ef"] = best_r
            closest_in_weights["var_ef"] = best_v
            closest_in_weights["sd_ef"] = sd_ef
            closest_in_weights["r_diff"] = best_r - r_w
            closest_in_weights["var_diff"] = best_v - var_w
            closest_in_weights["sd_diff"] = None if not sd else (sd_ef - sd_w)
            closest_in_weights["dissimilarity"] = float(best_D)
            closest_in_weights["w_ef"] = best_w.tolist()
            closest_in_weights["w_diff"] = w_diff.tolist()

    if verbose:
        print("[W-PORT] ef_same_var:", ef_same_var)
        print("[W-PORT] frontier_same_r:", frontier_same_r)
        print("[W-PORT] nearest_ef_point:", nearest_ef_point)
        print("[W-PORT] closest_in_weights:", closest_in_weights)

    return {
        "r_w": r_w,
        "var_w": var_w,
        "sd_w": sd_w if sd else None,
        "ef_same_var": ef_same_var,
        "frontier_same_r": frontier_same_r,
        "nearest_ef_point": nearest_ef_point,
        "closest_in_weights": closest_in_weights,
    }


# =========================
#   GRAPHING (AT THE END)
# =========================

def _plot_frontiers(mu, Sigma, segments, r_global,
                    sd=True, num_points_per_seg=200,
                    weights=None, w_diagnostics=None):
    """
    Transposed plot:
        x-axis  = sd (sqrt(var)) if sd=True, else variance
        y-axis  = expected return r

    If weights is not None, also plot:
        - the w portfolio (with legend entry)
        - the four diagnostic frontier portfolios (1–4), labeled in-plot.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    fig, ax = plt.subplots()

    def var_on_seg(seg, r):
        a = seg["a_scaled"]
        b = seg["b_scaled"]
        c = seg["c_scaled"]
        return a * r * r + b * r + c

    # --- 1) Assets (points above curves, but below diagnostic markers) ---
    asset_vars = np.diag(Sigma)
    if sd:
        asset_x = np.sqrt(np.maximum(asset_vars, 0.0))
    else:
        asset_x = asset_vars
    ax.scatter(asset_x, mu, marker='s', label='Assets', zorder=3)

    used_labels = set()

    # --- 2) Frontier segments (curves drawn with low zorder) ---
    for seg in segments:
        r_lo = seg["lower_r"]
        r_hi = seg["upper_r"]
        if r_hi <= r_lo:
            continue

        rs = np.linspace(r_lo, r_hi, num_points_per_seg)
        vs = var_on_seg(seg, rs)
        if sd:
            xs = np.sqrt(np.maximum(vs, 0.0))
        else:
            xs = vs

        if seg["ef_frontier"]:
            lbl = "NW EF"
            style = {"color": "blue", "lw": 2}
        elif seg["low_frontier"]:
            lbl = "SW frontier"
            style = {"color": "green", "lw": 2, "ls": "--"}
        elif seg["ea_frontier"]:
            lbl = "East frontier"
            style = {"color": "red", "lw": 2, "ls": ":"}
        else:
            lbl = None
            style = {}

        if lbl in used_labels:
            lbl = None
        else:
            if lbl:
                used_labels.add(lbl)

        ax.plot(xs, rs, label=lbl, zorder=1, **style)

    # --- 3) w portfolio (point with legend entry, above assets/curves) ---
    r_w = None
    var_w = None
    if weights is not None:
        w = np.asarray(weights, float)
        r_w = float(mu @ w)
        var_w = float(w @ (Sigma @ w))
        x_w = math.sqrt(max(var_w, 0.0)) if sd else var_w
        ax.scatter(x_w, r_w, marker='x', color='black',
                   label='Portfolio X', zorder=4)

    # --- 4) Diagnostic points 1–4 (on top of everything) ---
    if (weights is not None) and (w_diagnostics is not None):
        # helper to compute (x, y) from weights
        def xy_from_weights(w_star):
            r_star = float(mu @ w_star)
            var_star = float(w_star @ (Sigma @ w_star))
            x_star = math.sqrt(max(var_star, 0.0)) if sd else var_star
            return x_star, r_star

        # After all plotting so far, get axis ranges to scale label offsets
        fig.canvas.draw()  # ensure limits are up-to-date
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        dx = 0.01 * (x_max - x_min) if x_max > x_min else 0.0
        dy = 0.01 * (y_max - y_min) if y_max > y_min else 0.0

        ef_same_var = w_diagnostics.get("ef_same_var", {})
        fsr = w_diagnostics.get("frontier_same_r", {})
        nearest = w_diagnostics.get("nearest_ef_point", {})
        closest = w_diagnostics.get("closest_in_weights", {})

        # 1) same-variance frontier (label north)
        if ef_same_var.get("exists") and ef_same_var.get("w_frontier") is not None:
            w1 = np.asarray(ef_same_var["w_frontier"], float)
            x1, y1 = xy_from_weights(w1)
            ax.scatter(x1, y1, color='black', zorder=5)
            ax.text(x1, y1 + dy, "1",
                    color='black', fontsize=9,
                    ha='center', va='bottom', zorder=6)

        # 2) same-return frontier (label west/left)
        if fsr.get("exists") and fsr.get("w_frontier") is not None:
            w2 = np.asarray(fsr["w_frontier"], float)
            x2, y2 = xy_from_weights(w2)
            ax.scatter(x2, y2, color='black', zorder=5)
            ax.text(x2 - dx, y2, "2",
                    color='black', fontsize=9,
                    ha='right', va='center', zorder=6)

        # 3) nearest EF in (r,sd) space (label SE)
        if nearest.get("exists") and nearest.get("w_ef") is not None:
            w3 = np.asarray(nearest["w_ef"], float)
            x3, y3 = xy_from_weights(w3)
            ax.scatter(x3, y3, color='black', zorder=5)
            ax.text(x3 + dx, y3 - dy, "3",
                    color='black', fontsize=9,
                    ha='left', va='top', zorder=6)

        # 4) closest-in-weights EF (label NW)
        if closest.get("exists") and closest.get("w_ef") is not None:
            w4 = np.asarray(closest["w_ef"], float)
            x4, y4 = xy_from_weights(w4)
            ax.scatter(x4, y4, color='black', zorder=5)
            ax.text(x4 - dx, y4 + dy, "4",
                    color='black', fontsize=9,
                    ha='right', va='bottom', zorder=6)

    ax.set_xlabel("Standard deviation (σ)" if sd else "Variance (σ²)")
    ax.set_ylabel("Expected return (r)")
    ax.set_title("Markowitz Frontiers (Transposed)")
    ax.grid(True)
    ax.legend()
    plt.show()

# =========================
#   EXAMPLE USAGE
# =========================
#mu = np.array([0.2044, 0.1579, 0.095])
#mu = np.array([0.2044, 0.095, 0.1579])
#mu = np.array([0.1579, 0.2044, 0.095])
#mu = np.array([0.1579, 0.095, 0.2044])
#mu = np.array([0.095, 0.1579, 0.2044])
#mu = np.array([0.095, 0.2044, 0.1579])
#Sigma = np.array([
#    [0.00024086, 0.00005642, 0.00008801],
#    [0.00005642, 0.00011336, 0.00006400],
#    [0.00008801, 0.00006400, 0.00015271]
#])


#w = np.array([0.6,0.00,0.4])


if __name__ == "__main__":
    out = compute_frontiers(mu, Sigma, weights=w,ef_frontier=False, low_frontier=True, east_mode=True, east_K=200, verbose=False, sd=True, w_deltas=True, graph=True)

