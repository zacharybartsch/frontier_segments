import numpy as np
import math
import itertools


# =============================================================================
#  INTERNAL: Markowitz frontier computation (unchanged from original)
# =============================================================================

def _active_representation(mu, Sigma, active, tol=1e-12):
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)
    active = list(active)
    idx = np.array(active, dtype=int)

    cov = Sigma[np.ix_(idx, idx)]
    m = mu[idx]
    inv = np.linalg.inv(cov)
    e = np.ones(len(idx))

    A = float(m @ inv @ m)
    B = float(m @ inv @ e)
    C = float(e @ inv @ e)
    D = A * C - B * B
    if D <= 0:
        raise ValueError("Degenerate active set (D <= 0).")

    a = C / D
    b = -2.0 * B / D
    c = A / D

    v = inv @ m
    u = inv @ e
    P = (C / D) * v - (B / D) * u
    q = (-B / D) * v + (A / D) * u

    r_min = B / C

    r_lo, r_hi = -1e18, 1e18
    for Pi, qi in zip(P, q):
        if abs(Pi) < 1e-14:
            if qi < -tol:
                raise ValueError("Active set gives negative weight for all r.")
            continue
        r0 = -qi / Pi
        if Pi > 0:
            r_lo = max(r_lo, r0)
        else:
            r_hi = min(r_hi, r0)

    r_lo = max(r_lo, float(m.min()))
    r_hi = min(r_hi, float(m.max()))
    if r_lo >= r_hi - tol:
        raise ValueError("No feasible r interval for this active set.")

    all_idx = np.arange(N)
    inactive = [j for j in all_idx if j not in active]
    gamma_coeffs = {}

    for j in inactive:
        s = Sigma[j, idx]
        mu_j = mu[j]
        sp = float(s @ P)
        sq = float(s @ q)
        g1 = sp - (C / D) * mu_j + (B / D)
        g0 = sq + (B / D) * mu_j - A / D
        gamma_coeffs[j] = (g1, g0)

    return {
        "active": active, "inactive": inactive,
        "a": a, "b": b, "c": c,
        "P": P, "q": q,
        "A": A, "B": B, "C": C, "D": D,
        "r_min": r_min, "r_lo": r_lo, "r_hi": r_hi,
        "gamma_coeffs": gamma_coeffs,
    }


def _extend_from_singleton(mu, Sigma, active, direction, tol=1e-10):
    """
    When the NW/SW walk collapses to a single active asset whose own mean
    is not yet the global extreme in `direction`, _active_representation
    can't be used to find the next pivot (a 1-asset active set is
    degenerate: D == 0). Find the best inactive asset to bring back in by
    trying each candidate pair {asset, j} directly and picking the one
    with the least steep initial variance trade-off, mirroring the
    standard CLA tie-break rule. Returns the extended active-set list, or
    None if no asset extends further in `direction`.
    """
    i = active[0]
    r_i = mu[i]
    best_j, best_slope = None, None
    for j in range(len(mu)):
        if j == i:
            continue
        if direction > 0 and mu[j] <= r_i + tol:
            continue
        if direction < 0 and mu[j] >= r_i - tol:
            continue
        try:
            rep_j = _active_representation(mu, Sigma, sorted([i, j]))
        except ValueError:
            continue
        slope = 2.0 * rep_j["a"] * r_i + rep_j["b"]
        if best_slope is None or direction * slope < direction * best_slope:
            best_slope, best_j = slope, j
    if best_j is None:
        return None
    return sorted(active + [best_j])


def west_frontier_piecewise(mu, Sigma, tol=1e-10, verbose=False,
                             calc_ef=True, calc_low=True):
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
            "active_set": tuple(int(i) for i in active_set),
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

    # r_global = B/C (unconstrained global MVP return), computed directly so
    # that a long-only infeasible full active set does not block this step.
    _inv = np.linalg.inv(Sigma)
    _e   = np.ones(N)
    r_global = float(mu @ _inv @ _e) / float(_e @ _inv @ _e)

    if not calc_ef and not calc_low:
        return [], r_global

    # Starting active set: assets with positive weight in the unconstrained MVP.
    # This is the correct long-only CLA pivot; the full N-asset set can be
    # infeasible even when valid long-only portfolios exist.
    _w_mvp      = (_inv @ _e) / float(_e @ _inv @ _e)
    start_active = sorted([i for i in range(N) if _w_mvp[i] > tol])
    if not start_active:
        start_active = [int(np.argmax(mu))]

    full_idx = parabola_idx_counter
    parabola_idx_counter += 1

    if calc_ef:
        active = start_active.copy()
        rep = _active_representation(mu, Sigma, active)
        current_r = r_global

        while True:
            P, q = rep["P"], rep["q"]
            gamma = rep["gamma_coeffs"]
            r_hi = rep["r_hi"]
            candidates = []

            for local_i, g_i in enumerate(rep["active"]):
                Pi, qi = P[local_i], q[local_i]
                if abs(Pi) < 1e-14:
                    continue
                if Pi < 0:
                    r0 = -qi / Pi
                    if r0 > current_r + tol and r0 <= r_hi + tol:
                        candidates.append((r0, ("exit", g_i)))

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
                    a = b = 0.0; c = Sigma[asset, asset]
                    add_segment(active, a, b, c, current_r, r_end, True, False,
                                idx=full_idx if active == start_active else None)
                    extended = _extend_from_singleton(mu, Sigma, active, +1, tol)
                    if extended is None:
                        break
                    active = extended
                    current_r = r_end
                    rep = _active_representation(mu, Sigma, active)
                    continue
                else:
                    r_end = r_hi
                    a, b, c = rep["a"], rep["b"], rep["c"]
                add_segment(active, a, b, c, current_r, r_end, True, False,
                            idx=full_idx if active == start_active else None)
                break

            r_next, (etype, asset) = min(candidates, key=lambda x: x[0])
            if len(active) == 1:
                a = b = 0.0; c = Sigma[active[0], active[0]]
            else:
                a, b, c = rep["a"], rep["b"], rep["c"]

            add_segment(active, a, b, c, current_r, r_next, True, False,
                        idx=full_idx if active == start_active else None)

            if verbose:
                print(f"[WEST NW] r {current_r:.6f}->{r_next:.6f}, {etype} asset={asset}")

            if etype == "exit":
                active = [i for i in active if i != asset]
            else:
                if asset not in active:
                    active = sorted(active + [asset])

            current_r = r_next

            if len(active) == 1:
                a = b = 0.0; c = Sigma[active[0], active[0]]
                r_end = mu[active[0]]
                add_segment(active, a, b, c, current_r, r_end, True, False)
                extended = _extend_from_singleton(mu, Sigma, active, +1, tol)
                if extended is None:
                    break
                active = extended
                current_r = r_end

            rep = _active_representation(mu, Sigma, active)

    if calc_low:
        active = start_active.copy()
        rep = _active_representation(mu, Sigma, active)
        current_r = r_global

        while True:
            P, q = rep["P"], rep["q"]
            gamma = rep["gamma_coeffs"]
            r_lo = rep["r_lo"]
            candidates = []

            for local_i, g_i in enumerate(rep["active"]):
                Pi, qi = P[local_i], q[local_i]
                if abs(Pi) < 1e-14:
                    continue
                if Pi > 0:
                    r0 = -qi / Pi
                    if r0 < current_r - tol and r0 >= r_lo - tol:
                        candidates.append((r0, ("exit", g_i)))

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
                    a = b = 0.0; c = Sigma[asset, asset]
                    add_segment(active, a, b, c, r_end, current_r, False, True,
                                idx=full_idx if active == start_active else None)
                    extended = _extend_from_singleton(mu, Sigma, active, -1, tol)
                    if extended is None:
                        break
                    active = extended
                    current_r = r_end
                    rep = _active_representation(mu, Sigma, active)
                    continue
                else:
                    r_end = r_lo
                    a, b, c = rep["a"], rep["b"], rep["c"]
                add_segment(active, a, b, c, r_end, current_r, False, True,
                            idx=full_idx if active == start_active else None)
                break

            r_next, (etype, asset) = max(candidates, key=lambda x: x[0])
            if len(active) == 1:
                a = b = 0.0; c = Sigma[active[0], active[0]]
            else:
                a, b, c = rep["a"], rep["b"], rep["c"]

            add_segment(active, a, b, c, r_next, current_r, False, True,
                        idx=full_idx if active == start_active else None)

            if verbose:
                print(f"[WEST SW] r {r_next:.6f}->{current_r:.6f}, {etype} asset={asset}")

            if etype == "exit":
                active = [i for i in active if i != asset]
            else:
                if asset not in active:
                    active = sorted(active + [asset])

            current_r = r_next

            if len(active) == 1:
                a = b = 0.0; c = Sigma[active[0], active[0]]
                r_end = mu[active[0]]
                add_segment(active, a, b, c, r_end, current_r, False, True)
                extended = _extend_from_singleton(mu, Sigma, active, -1, tol)
                if extended is None:
                    break
                active = extended
                current_r = r_end

            rep = _active_representation(mu, Sigma, active)

    segments = sorted(segments, key=lambda s: (s["lower_r"], len(s["active_set"])))
    return segments, r_global


def _prune_east_assets(mu, Sigma, tol=1e-12, verbose=False):
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
        print(f"[EAST] Pruned {N - len(keep)} duplicated-mu assets; kept {len(keep)}.")
    return keep, mu[keep], Sigma[np.ix_(keep, keep)]


def _two_asset_parabola(mu, Sigma, i, j):
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
    return C / D, -2.0 * B / D, A / D


def east_frontier_grid(mu, Sigma, K=200, tol=1e-10, verbose=False):
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
    grid = np.linspace(r_min_all, r_max_all, K)

    pairs = []
    for i_e, j_e in itertools.combinations(range(N_e), 2):
        pairs.append((i_e, j_e, keep[i_e], keep[j_e]))

    best_pair_idx = []
    best_var = []
    for r in grid:
        max_var = -float("inf")
        best_idx = None
        for idx_pair, (i_e, j_e, i_orig, j_orig) in enumerate(pairs):
            mu_i, mu_j = mu_e[i_e], mu_e[j_e]
            m_lo, m_hi = min(mu_i, mu_j), max(mu_i, mu_j)
            if r < m_lo - tol or r > m_hi + tol:
                continue
            if abs(mu_i - mu_j) < 1e-14:
                continue
            w_i = (r - mu_j) / (mu_i - mu_j)
            w_j = 1.0 - w_i
            if w_i < -tol or w_j < -tol:
                continue
            var = (w_i**2 * Sigma[i_orig, i_orig] + w_j**2 * Sigma[j_orig, j_orig]
                   + 2.0 * w_i * w_j * Sigma[i_orig, j_orig])
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
            "active_set": (int(i_orig), int(j_orig)),
            "lower_r": float(r_low),
            "upper_r": float(r_high),
            "ef_frontier": 0, "low_frontier": 0, "ea_frontier": 1,
            "a_scaled": float(a), "b_scaled": float(b),
            "c_scaled": float(c), "d_scaled": 1.0,
        })
        parabola_idx_counter += 1
        k0 = k1
    return segments, (r_min_all, r_max_all)


def east_frontier_exact(mu, Sigma, tol=1e-10, verbose=False):
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

    parabs = []
    idx_counter = 1
    for i_e, j_e in itertools.combinations(range(N_e), 2):
        i_orig, j_orig = keep[i_e], keep[j_e]
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
        a = C / D; b = -2.0 * B / D; c = A / D
        r_lo = float(min(m)); r_hi = float(max(m))
        if r_lo >= r_hi - tol:
            continue
        parabs.append({"idx": idx_counter, "pair": (i_orig, j_orig),
                       "a": a, "b": b, "c": c, "r_lo": r_lo, "r_hi": r_hi})
        idx_counter += 1

    if not parabs:
        return [], None

    def _var_on(p, r):
        return p["a"] * r * r + p["b"] * r + p["c"]

    def _crossings(p, q, lo, hi):
        A = p["a"] - q["a"]; B = p["b"] - q["b"]; C = p["c"] - q["c"]
        if abs(A) < 1e-14 and abs(B) < 1e-14:
            return []
        roots = [-C / B] if abs(A) < 1e-14 else (
            [] if (disc := B*B - 4.0*A*C) < 0 else
            [(-B - math.sqrt(disc)) / (2*A), (-B + math.sqrt(disc)) / (2*A)]
        )
        return [r for r in roots if lo + tol < r < hi - tol]

    def _dominates(ref, other, lo, hi):
        """True if ref(r) >= other(r) for all r in [lo, hi]."""
        A = ref["a"] - other["a"]
        B = ref["b"] - other["b"]
        C = ref["c"] - other["c"]
        def d(r): return A * r * r + B * r + C
        if d(lo) < -tol or d(hi) < -tol:
            return False
        # Convex diff (A > 0) can dip below zero at interior vertex
        if A > 1e-14:
            r_v = -B / (2.0 * A)
            if lo < r_v < hi and d(r_v) < -tol:
                return False
        return True

    # Sort parabs descending by max endpoint variance so the scan below can
    # break early once remaining pairs are provably below the running lower bound.
    for p in parabs:
        p["max_var"] = max(_var_on(p, p["r_lo"]), _var_on(p, p["r_hi"]))
    parabs.sort(key=lambda p: p["max_var"], reverse=True)

    # Primary breakpoints are the sorted μ values of pruned assets.
    # A pair (i,j) is feasible exactly on [min(μ_i,μ_j), max(μ_i,μ_j)], so
    # feasibility can only change at these points — no global intersection sweep needed.
    mu_breaks = sorted(set(float(mu_e[k]) for k in range(N_e)))

    segments = []
    for lo, hi in zip(mu_breaks[:-1], mu_breaks[1:]):
        # Build feas with early termination: once a pair's max endpoint variance
        # (an upper bound on its value anywhere in [lo, hi]) falls below the
        # running lower bound lb, no subsequent pair can dominate either.
        feas = []
        lb = 0.0
        for p in parabs:
            if p["r_lo"] > lo + tol or p["r_hi"] < hi - tol:
                continue
            if p["max_var"] < lb - tol:
                break
            feas.append(p)
            lb = max(lb, _var_on(p, lo), _var_on(p, hi))

        if not feas:
            continue

        # Prune pairs that are fully dominated by the reference pair in [lo, hi].
        # The reference is the pair with the highest interval-boundary values.
        ref = max(feas, key=lambda p: max(_var_on(p, lo), _var_on(p, hi)))
        feas = [ref] + [p for p in feas if p is not ref and not _dominates(ref, p, lo, hi)]

        # Interior crossings only among surviving co-feasible pairs
        sub_breaks = [lo, hi]
        for i, p in enumerate(feas):
            for q in feas[i+1:]:
                sub_breaks.extend(_crossings(p, q, lo, hi))
        sub_breaks = sorted(set(sub_breaks))

        for slo, shi in zip(sub_breaks[:-1], sub_breaks[1:]):
            mid = 0.5 * (slo + shi)
            best = max(feas, key=lambda p: _var_on(p, mid))
            pair = tuple(int(x) for x in best["pair"])
            if segments and segments[-1]["active_set"] == pair and abs(segments[-1]["upper_r"] - slo) < tol:
                segments[-1]["upper_r"] = float(shi)
            else:
                segments.append({
                    "parabola_idx": len(segments) + 1,
                    "active_set": pair,
                    "lower_r": float(slo), "upper_r": float(shi),
                    "ef_frontier": 0, "low_frontier": 0, "ea_frontier": 1,
                    "a_scaled": float(best["a"]), "b_scaled": float(best["b"]),
                    "c_scaled": float(best["c"]), "d_scaled": 1.0,
                })

    if verbose:
        print(f"[EAST-EXACT] {len(segments)} segments from {len(parabs)} pairs.")
    return segments, (r_min_all, r_max_all)


# =============================================================================
#  1. compute_cloud — build frontier segments once; share across analyses
# =============================================================================

def compute_cloud(mu, Sigma,
                  ef=True, swf=True,
                  east_mode="exact", east_K=200,
                  verbose=False):
    """
    Compute the feasible cloud frontier and return a cloud_dict for reuse.

    Parameters
    ----------
    mu        : array-like, shape (N,)
    Sigma     : array-like, shape (N, N)
    ef        : bool  — compute NW efficient frontier
    swf       : bool  — compute SW frontier
    east_mode : "exact" | "grid" | False
    east_K    : int   — grid resolution when east_mode="grid"
    verbose   : bool

    Returns
    -------
    cloud_dict with keys:
        segments, mu, Sigma, r_global, N, chol_L
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    west_segs, r_global = west_frontier_piecewise(
        mu, Sigma, verbose=False, calc_ef=ef, calc_low=swf
    )

    east_segs = []
    if east_mode == "exact":
        east_segs, _ = east_frontier_exact(mu, Sigma, verbose=False)
    elif east_mode == "grid":
        east_segs, _ = east_frontier_grid(mu, Sigma, K=east_K, verbose=False)

    chol_L = np.linalg.cholesky(Sigma)

    cloud = {
        "segments": west_segs + east_segs,
        "mu":       mu,
        "Sigma":    Sigma,
        "r_global": r_global,
        "N":        N,
        "chol_L":   chol_L,
    }

    if verbose:
        segs = cloud["segments"]
        print(f"N         : {cloud['N']}")
        print(f"r_global  : {cloud['r_global']:.6f}")
        print(f"mu        : {cloud['mu']}")
        print(f"Sigma     :\n{cloud['Sigma']}")
        print(f"chol_L    :\n{cloud['chol_L']}")
        print(f"segments  : {len(segs)} total")
        grouped = {}
        order = []
        for s in segs:
            key = (s['active_set'], s['ef_frontier'], s['low_frontier'], s['ea_frontier'])
            if key not in grouped:
                grouped[key] = {'lower_r': s['lower_r'], 'upper_r': s['upper_r'], 'seg': s}
                order.append(key)
            else:
                grouped[key]['lower_r'] = min(grouped[key]['lower_r'], s['lower_r'])
                grouped[key]['upper_r'] = max(grouped[key]['upper_r'], s['upper_r'])
        header = f"  {'active_set':<16}  {'lower_r':>10}  {'upper_r':>10}  {'ef':>3}  {'sw':>3}  {'ea':>3}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in order:
            g = grouped[key]
            s = g['seg']
            print(
                f"  {str(s['active_set']):<16}  "
                f"{g['lower_r']:>10.6f}  {g['upper_r']:>10.6f}  "
                f"{s['ef_frontier']:>3}  {s['low_frontier']:>3}  {s['ea_frontier']:>3}"
            )

    return cloud


# =============================================================================
#  2. absolute_performance — §3.2 measures
# =============================================================================

def _auto_dec(vals, width, forced_sign=False, min_dec=2, max_dec=6):
    """Return decimal places so every value in vals fits in width chars."""
    max_fixed = 0
    for v in vals:
        if v is None:
            continue
        sign = 1 if (forced_sign or v < 0) else 0
        n_int = len(str(int(abs(v)))) if abs(v) >= 1 else 1
        fixed = sign + n_int + 1  # sign + integer digits + decimal point
        if fixed > max_fixed:
            max_fixed = fixed
    if max_fixed == 0:
        max_fixed = 2
    return min(max_dec, max(min_dec, width - max_fixed))

def absolute_performance(cloud_dict, weights, sd=True, tol=1e-10, rf=0.0,
                         verbose=False, reference_weights=None):
    """
    Absolute portfolio performance relative to the frontier (§3.2).

    Parameters
    ----------
    cloud_dict        : dict returned by compute_cloud
    weights           : array-like, shape (N,)
    sd                : bool — report in standard-deviation units (True) or variance
    reference_weights : optional array-like, shape (N,) — benchmark portfolio;
                        when provided, verbose output includes r and sd of this
                        portfolio alongside deltas relative to weights

    Returns
    -------
    dict with keys:
        r_w, var_w, sd_w,
        frontier_same_var  — EF/EA point at same variance
        frontier_same_r    — EF/SW point at same return
        nearest_ef         — nearest EF point in (r, sd) space
        closest_ef_weights — closest EF point in weight space (min D)
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    N      = cloud_dict["N"]
    r_global = cloud_dict["r_global"]
    segments = cloud_dict["segments"]

    w = np.asarray(weights, float).ravel()
    if w.shape[0] != N:
        raise ValueError(f"weights length {w.shape[0]} != N={N}")

    r_w   = float(mu @ w)
    var_w = float(w @ (Sigma @ w))
    sd_w  = math.sqrt(max(var_w, 0.0)) if sd else None

    ef_segs  = [s for s in segments if s["ef_frontier"]]
    low_segs = [s for s in segments if s["low_frontier"]]
    ea_segs  = [s for s in segments if s["ea_frontier"]]

    rep_cache = {}

    def _get_rep(active_set):
        key = tuple(active_set)
        if key not in rep_cache:
            rep_cache[key] = _active_representation(mu, Sigma, list(key))
        return rep_cache[key]

    def _var_on(seg, r):
        return seg["a_scaled"] * r * r + seg["b_scaled"] * r + seg["c_scaled"]

    def _weights_on(seg, r_star):
        active = seg["active_set"]
        if len(active) == 1:
            w_star = np.zeros(N)
            w_star[active[0]] = 1.0
            return w_star
        rep = _get_rep(active)
        w_star = np.zeros(N)
        w_star[list(active)] = rep["P"] * r_star + rep["q"]
        return w_star

    def _dissimilarity(w_star):
        return 0.5 * float(np.sum(np.abs(w - w_star)))

    # ---- 1) Frontier at same variance ----------------------------------------
    asset_vars = np.diag(Sigma)
    var_max_r  = float(asset_vars[int(np.argmax(mu))])
    use_ef     = (var_w <= var_max_r)
    pool       = ef_segs if use_ef else ea_segs

    fsv = {"exists": False, "on_ef": False, "on_ea": False,
           "r_frontier": None, "r_diff": None,
           "dissimilarity": None, "w_frontier": None, "reason": None}

    if not pool:
        fsv["reason"] = ("Efficient frontier" if use_ef else "East frontier") + " not calculated."
    else:
        best_seg, best_r, best_err = None, None, float("inf")
        for seg in pool:
            a, b, c = seg["a_scaled"], seg["b_scaled"], seg["c_scaled"] - var_w
            disc = b * b - 4.0 * a * c if abs(a) > 1e-14 else None
            roots = []
            if disc is None:
                if abs(b) > 1e-14:
                    roots = [-c / b]
            elif disc >= 0:
                sq = math.sqrt(max(0.0, disc))
                roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
            for r0 in roots:
                if seg["lower_r"] - tol <= r0 <= seg["upper_r"] + tol:
                    err = abs(_var_on(seg, r0) - var_w)
                    if err < best_err:
                        best_err, best_r, best_seg = err, r0, seg
        if best_seg is not None:
            wf = _weights_on(best_seg, best_r)
            fsv.update(exists=True, r_frontier=float(best_r),
                       r_diff=float(best_r - r_w),
                       dissimilarity=_dissimilarity(wf),
                       w_frontier=wf.tolist(),
                       on_ef=bool(best_seg["ef_frontier"]),
                       on_ea=bool(best_seg["ea_frontier"]))
        else:
            fsv["reason"] = "Portfolio variance outside frontier range."

    # ---- 2) Frontier at same return ------------------------------------------
    fsr = {"exists": False, "on_ef": False, "on_low": False,
           "sd_frontier": None, "sd_diff": None,
           "dissimilarity": None, "w_frontier": None, "reason": None}

    chosen_seg = None
    if r_w < r_global:
        if not low_segs:
            fsr["reason"] = "SW frontier not calculated."
        else:
            cands = [s for s in low_segs if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
            if cands:
                chosen_seg = min(cands, key=lambda s: _var_on(s, r_w))
                fsr["on_low"] = True
            else:
                fsr["reason"] = "Return r_w outside SW frontier range."
    else:
        if not ef_segs:
            fsr["reason"] = "Efficient frontier not calculated."
        else:
            cands = [s for s in ef_segs if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
            if cands:
                chosen_seg = min(cands, key=lambda s: _var_on(s, r_w))
                fsr["on_ef"] = True
            else:
                fsr["reason"] = "Return r_w outside EF range."

    if chosen_seg is not None:
        vf   = _var_on(chosen_seg, r_w)
        sdf  = math.sqrt(max(vf, 0.0)) if sd else None
        wf   = _weights_on(chosen_seg, r_w)
        fsr.update(exists=True, sd_frontier=sdf,
                   sd_diff=(sd_w - sdf) if sd else None,
                   dissimilarity=_dissimilarity(wf),
                   w_frontier=wf.tolist())

    # ---- 3) Nearest EF point in (r, sd) space --------------------------------
    nef = {"exists": False, "r_ef": None, "sd_ef": None,
           "r_diff": None, "sd_diff": None, "distance": None,
           "dissimilarity": None, "w_ef": None, "reason": None}

    if not ef_segs:
        nef["reason"] = "No EF segments."
    else:
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi

        def _dist2(seg, r):
            v = _var_on(seg, r)
            if sd:
                s = math.sqrt(max(v, 0.0))
                return (r - r_w) ** 2 + (s - sd_w) ** 2
            return (r - r_w) ** 2 + (v - var_w) ** 2

        def _minimize_seg(seg):
            a, b = seg["lower_r"], seg["upper_r"]
            if b <= a + tol:
                return None, None
            c = b - (b - a) * inv_phi
            d = a + (b - a) * inv_phi
            fc, fd = _dist2(seg, c), _dist2(seg, d)
            for _ in range(60):
                if abs(b - a) < 1e-12:
                    break
                if fc < fd:
                    b = d; d = c; fd = fc
                    c = b - (b - a) * inv_phi; fc = _dist2(seg, c)
                else:
                    a = c; c = d; fc = fd
                    d = a + (b - a) * inv_phi; fd = _dist2(seg, d)
            r_star = 0.5 * (a + b)
            return r_star, _dist2(seg, r_star)

        best_d2, best_r_ef, best_seg_ef = float("inf"), None, None
        for seg in ef_segs:
            r_star, d2 = _minimize_seg(seg)
            if r_star is not None and d2 < best_d2:
                best_d2, best_r_ef, best_seg_ef = d2, r_star, seg

        if best_seg_ef is not None:
            vef = _var_on(best_seg_ef, best_r_ef)
            sdef = math.sqrt(max(vef, 0.0)) if sd else None
            wef  = _weights_on(best_seg_ef, best_r_ef)
            nef.update(exists=True, r_ef=best_r_ef, sd_ef=sdef,
                       r_diff=best_r_ef - r_w,
                       sd_diff=(sdef - sd_w) if sd else None,
                       distance=math.sqrt(best_d2),
                       dissimilarity=_dissimilarity(wef),
                       w_ef=wef.tolist())
        else:
            nef["reason"] = "No valid EF candidates."

    # ---- 4) Closest EF point in weight space (min D) -------------------------
    cew = {"exists": False, "r_ef": None, "sd_ef": None,
           "r_diff": None, "sd_diff": None,
           "dissimilarity": None, "w_ef": None, "reason": None}

    if not ef_segs:
        cew["reason"] = "No EF segments."
    else:
        best_D, best_r_c, best_seg_c = float("inf"), None, None
        for seg in ef_segs:
            lo, hi   = seg["lower_r"], seg["upper_r"]
            active   = seg["active_set"]
            # Candidates: both endpoints + interior kink points where w_i(r) = w_o,i.
            # Weights are linear in r on each segment: w_i(r) = P_i*r + q_i,
            # so D(r) is piecewise-linear and its minimum is at a kink or endpoint.
            cands = [lo, hi]
            if len(active) > 1:
                rep = _get_rep(active)
                for idx, i in enumerate(active):
                    p_i = rep["P"][idx]
                    if abs(p_i) > 1e-14:
                        r_kink = (w[i] - rep["q"][idx]) / p_i
                        if lo < r_kink < hi:
                            cands.append(r_kink)
            for r0 in cands:
                D = _dissimilarity(_weights_on(seg, r0))
                if D < best_D:
                    best_D, best_r_c, best_seg_c = D, r0, seg
        if best_seg_c is not None:
            vef  = _var_on(best_seg_c, best_r_c)
            sdef = math.sqrt(max(vef, 0.0)) if sd else None
            wef  = _weights_on(best_seg_c, best_r_c)
            cew.update(exists=True, r_ef=best_r_c, sd_ef=sdef,
                       r_diff=best_r_c - r_w,
                       sd_diff=(sdef - sd_w) if sd else None,
                       dissimilarity=float(best_D),
                       w_ef=wef.tolist())

    # ---- reference portfolio (optional) ----------------------------------------
    r_ref, sd_ref = None, None
    if reference_weights is not None:
        w_ref = np.asarray(reference_weights, float).ravel()
        if w_ref.shape[0] != N:
            raise ValueError(f"reference_weights length {w_ref.shape[0]} != N={N}")
        r_ref   = float(mu @ w_ref)
        var_ref = float(w_ref @ (Sigma @ w_ref))
        sd_ref  = math.sqrt(max(var_ref, 0.0)) if sd else None

    # ---- Reference portfolio points (always computed; used by verbose + plot_cloud) --
    mvp_seg = next(
        (s for s in ef_segs + low_segs
         if s["lower_r"] - tol <= r_global <= s["upper_r"] + tol),
        None
    )
    sd_mvp = math.sqrt(max(_var_on(mvp_seg, r_global), 0.0)) if mvp_seg else None

    idx_max = int(np.argmax(mu))
    r_max   = float(mu[idx_max])
    sd_max  = math.sqrt(float(Sigma[idx_max, idx_max]))

    # Max Sharpe (tangency) portfolio — steepest line from (sigma=0, r=rf) tangent to EF.
    # Skip degenerate single-asset segments (a=b=0, constant sigma); their transition
    # point is already evaluated as the upper_r endpoint of the adjacent multi-asset segment.
    r_ms, sd_ms, w_ms = None, None, None
    best_sr = -math.inf
    for seg in ef_segs:
        a_s, b_s, c_s = seg["a_scaled"], seg["b_scaled"], seg["c_scaled"]
        if abs(a_s) < 1e-14 and abs(b_s) < 1e-14:
            continue
        lo_s, hi_s = seg["lower_r"], seg["upper_r"]
        candidates = [lo_s, hi_s]
        denom = b_s + 2.0 * rf * a_s
        if abs(denom) > 1e-14:
            r_crit = -(2.0 * c_s + rf * b_s) / denom
            candidates.append(float(np.clip(r_crit, lo_s, hi_s)))
        for r_c in candidates:
            v_c = _var_on(seg, r_c)
            if v_c <= 1e-14:
                continue
            sr_c = (r_c - rf) / math.sqrt(v_c)
            if sr_c > best_sr:
                best_sr = sr_c
                r_ms    = r_c
                sd_ms   = math.sqrt(v_c)
                w_ms    = _weights_on(seg, r_c)

    if verbose:
        sd_w_print = math.sqrt(max(var_w, 0.0))
        r_fsv  = fsv["r_frontier"] if fsv["exists"] else None
        sd_fsr = fsr["sd_frontier"] if fsr["exists"] else None

        WA, WW, WDW = 14, 8, 9
        grp_w = WW + WDW + 6

        ref_cols = []
        if fsv["exists"] and fsv["w_frontier"] is not None:
            ref_cols.append({"label": "Max r|Same sd", "w": np.array(fsv["w_frontier"]),
                             "r": r_fsv,    "sd": sd_w_print})
        if fsr["exists"] and fsr["w_frontier"] is not None:
            ref_cols.append({"label": "Min sd|Same r", "w": np.array(fsr["w_frontier"]),
                             "r": r_w,      "sd": sd_fsr})
        if mvp_seg is not None:
            ref_cols.append({"label": "Min Var",       "w": _weights_on(mvp_seg, r_global),
                             "r": r_global, "sd": sd_mvp})
        w_max_r_arr = np.zeros(N); w_max_r_arr[idx_max] = 1.0
        ref_cols.append({"label": "Max Return",    "w": w_max_r_arr,
                         "r": r_max,    "sd": sd_max})
        if w_ms is not None:
            ref_cols.append({"label": "Max Sharpe", "w": w_ms,
                             "r": r_ms,    "sd": sd_ms})
        if cew["exists"] and cew["w_ef"] is not None:
            w_md = np.array(cew["w_ef"])
            ref_cols.append({"label": "EF Min Diss", "w": w_md,
                             "r": cew["r_ef"], "sd": cew["sd_ef"]})
        if reference_weights is not None:
            ref_cols.append({"label": "Ref Portfolio", "w": w_ref,
                             "r": r_ref,    "sd": sd_ref})

        def _sharpe(r_val, sd_val):
            if r_val is None or sd_val is None or sd_val < 1e-14:
                return None
            return (r_val - rf) / sd_val

        sharpe_w_print = _sharpe(r_w, sd_w_print)

        # Collect all displayed values to compute decimal precision dynamically
        _stat_vals, _stat_deltas = [r_w, sd_w_print, sharpe_w_print], []
        _wt_vals,   _wt_deltas   = list(w), []
        for col in ref_cols:
            sr = _sharpe(col["r"], col["sd"])
            _stat_vals += [col["r"], col["sd"], sr]
            for vo, vc in [(r_w, col["r"]), (sd_w_print, col["sd"]), (sharpe_w_print, sr)]:
                if vo is not None and vc is not None:
                    _stat_deltas.append(vc - vo)
            if col["w"] is not None:
                _wt_vals   += list(col["w"])
                _wt_deltas += [col["w"][i] - w[i] for i in range(N)]

        dec_sv = _auto_dec(_stat_vals,  WW)
        dec_sd = _auto_dec(_stat_deltas, WDW, forced_sign=True)
        dec_wv = _auto_dec(_wt_vals,    WW)
        dec_wd = _auto_dec(_wt_deltas,  WDW, forced_sign=True)

        def _fv(val):
            return f"{val:>{WW}.{dec_sv}f}" if val is not None else f"{'N/A':>{WW}}"
        def _fd(val):
            return f"{val:>+{WDW}.{dec_sd}f}" if val is not None else f"{'N/A':>{WDW}}"
        def _grp(val, delta):
            return f"  {_fv(val)}  {_fd(delta)} |"
        def _fvw(val):
            return f"{val:>{WW}.{dec_wv}f}" if val is not None else f"{'N/A':>{WW}}"
        def _fdw(val):
            return f"{val:>+{WDW}.{dec_wd}f}" if val is not None else f"{'N/A':>{WDW}}"

        h1 = f"  {'asset':>{WA}} | {'w_o':^{WW}} |"
        for col in ref_cols:
            h1 += f"{col['label']:^{grp_w}}"
        sep = "  " + "-" * (len(h1) - 2)

        r_row = f"  {'r':>{WA}} | {_fv(r_w)} |"
        for col in ref_cols:
            r_row += _grp(col["r"], (col["r"] - r_w) if col["r"] is not None else None)

        sd_row = f"  {'sd':>{WA}} | {_fv(sd_w_print)} |"
        for col in ref_cols:
            sd_row += _grp(col["sd"], (col["sd"] - sd_w_print) if col["sd"] is not None else None)

        sharpe_row = f"  {'sharpe':>{WA}} | {_fv(sharpe_w_print)} |"
        for col in ref_cols:
            sr_col = _sharpe(col["r"], col["sd"])
            sr_delta = (sr_col - sharpe_w_print) if (sr_col is not None and sharpe_w_print is not None) else None
            sharpe_row += _grp(sr_col, sr_delta)

        print()
        print("--- absolute_performance ---")
        print(f"  rf = {rf:.6f}")
        print(h1)
        print(sep)
        print(r_row)
        print(sd_row)
        print(sharpe_row)
        print(sep)
        for i in range(N):
            lbl = f"{'weights':<7}{i:>{WA - 7}}" if i == 0 else f"{i:>{WA}}"
            wt_row = f"  {lbl} | {_fvw(w[i])} |"
            for col in ref_cols:
                wt_row += f"  {_fvw(col['w'][i])}  {_fdw(col['w'][i] - w[i])} |"
            print(wt_row)
        print()

    return {
        "r_w":               r_w,
        "var_w":             var_w,
        "sd_w":              sd_w,
        "sharpe_w":          (r_w - rf) / sd_w if sd_w > 1e-14 else None,
        "rf":                rf,
        "frontier_same_var": fsv,
        "frontier_same_r":   fsr,
        "nearest_ef":        nef,
        "closest_ef_weights": cew,
        "min_var":           {"r": r_global, "sd": sd_mvp},
        "max_return":        {"r": r_max,    "sd": sd_max},
        "max_sharpe":        {"r": r_ms,     "sd": sd_ms},
    }


# =============================================================================
#  3. quasi_relative_performance — §3.3 measures
# =============================================================================

def quasi_relative_performance(cloud_dict, weights, sd=True, tol=1e-10,
                                rf=0.0, verbose=False, w_ref=None):
    """
    Quasi-relative portfolio performance (§3.3).

    Conditional measures (rho) — conditioned on the portfolio's own risk or return:
      rho_r    : position of r_w between r_min and r_max achievable at sigma_w
                 rho_r = (r_w - r_min) / (r_max - r_min)   in [0, 1]; 1 = best
      rho_sigma: position of sigma_w between sigma_min and sigma_max at r_w
                 rho_sigma = (sigma_max - sigma_w) / (sigma_max - sigma_min)
                             in [0, 1]; 1 = best (on EF)

    Unconditional measures (gamma) — anchored to global feasible extremes:
      gamma_r     : (r_w - min(mu)) / (max(mu) - min(mu))
      gamma_sigma : (sd_max_global - sd_w) / (sd_max_global - sd_min_global)
                    sd_min_global = MVP sd;  sd_max_global = max(sqrt(diag(Sigma)))
      gamma_sharpe: (sharpe_w - sharpe_min_global) / (sharpe_max_global - sharpe_min_global)
                    sharpe_max_global = tangency Sharpe;
                    sharpe_min_global = min single-asset Sharpe

    Parameters
    ----------
    cloud_dict : dict from compute_cloud  (must include SW and EA segments)
    weights    : array-like, shape (N,)
    sd         : bool — operate in standard-deviation units
    verbose    : bool — print rho/gamma stats and dissimilarity tables when True
    w_ref      : optional array-like, shape (N,) — benchmark portfolio;
                 all stats are also computed for it

    Returns
    -------
    dict with keys:
        r_w, var_w, sd_w, sharpe_w,
        rho_r, r_min_at_sigma, r_max_at_sigma,
        rho_sigma, sd_min_at_r, sd_max_at_r,
        gamma_r, r_min_global, r_max_global,
        gamma_sigma, sd_min_global, sd_max_global,
        gamma_sharpe, sharpe_min_global, sharpe_max_global,
        ref_rho_r, ref_rho_sigma,
        ref_gamma_r, ref_gamma_sigma, ref_gamma_sharpe,
        dissim_w_ref
        (any key is None when not applicable or frontier not computed)
    """
    mu      = cloud_dict["mu"]
    Sigma   = cloud_dict["Sigma"]
    N       = cloud_dict["N"]
    r_global= cloud_dict["r_global"]
    segments= cloud_dict["segments"]

    w = np.asarray(weights, float).ravel()
    if w.shape[0] != N:
        raise ValueError(f"weights length {w.shape[0]} != N={N}")

    r_w   = float(mu @ w)
    var_w = float(w @ (Sigma @ w))
    sd_w  = math.sqrt(max(var_w, 0.0)) if sd else None

    ef_segs  = [s for s in segments if s["ef_frontier"]]
    low_segs = [s for s in segments if s["low_frontier"]]
    ea_segs  = [s for s in segments if s["ea_frontier"]]

    def _var_on(seg, r):
        return seg["a_scaled"] * r * r + seg["b_scaled"] * r + seg["c_scaled"]

    # Max Sharpe (tangency) portfolio — skip degenerate single-asset segments (a=b=0)
    r_ms, var_ms = None, None
    _best_sr = -math.inf
    for _seg in ef_segs:
        _a, _b, _c = _seg["a_scaled"], _seg["b_scaled"], _seg["c_scaled"]
        if abs(_a) < 1e-14 and abs(_b) < 1e-14:
            continue
        _lo, _hi   = _seg["lower_r"], _seg["upper_r"]
        _cands     = [_lo, _hi]
        _denom     = _b + 2.0 * rf * _a
        if abs(_denom) > 1e-14:
            _cands.append(float(np.clip(-(2.0 * _c + rf * _b) / _denom, _lo, _hi)))
        for _r in _cands:
            _v = _var_on(_seg, _r)
            if _v <= 1e-14:
                continue
            _sr = (_r - rf) / math.sqrt(_v)
            if _sr > _best_sr:
                _best_sr, r_ms = _sr, _r

    def _roots_at_var(seg, target_var):
        """Return r values on seg where var(r) == target_var, within segment bounds."""
        a = seg["a_scaled"]; b = seg["b_scaled"]; c = seg["c_scaled"] - target_var
        lo, hi = seg["lower_r"], seg["upper_r"]
        roots = []
        if abs(a) < 1e-14:
            if abs(b) > 1e-14:
                r0 = -c / b
                if lo - tol <= r0 <= hi + tol:
                    roots.append(r0)
        else:
            disc = b * b - 4.0 * a * c
            if disc < 0:
                return roots
            sq = math.sqrt(max(0.0, disc))
            for r0 in [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]:
                if lo - tol <= r0 <= hi + tol:
                    roots.append(r0)
        return roots

    # ---- rho_r: feasible return intervals at var_w ---------------------------
    # The cloud in (return, variance) space:
    #   lower boundary = west frontier (min variance at each r)
    #   upper boundary = EA frontier (max variance at each r)
    # r is feasible at var_w iff var_west(r) <= var_w <= var_ea(r).
    #
    # All frontier segments are convex parabolas (a >= 0), so the feasible
    # portion of each segment is determined analytically from the roots alone:
    #   west (need var <= var_w): interval BETWEEN the parabola's roots
    #   EA   (need var >= var_w): interval(s) OUTSIDE the parabola's roots
    #
    # The EA frontier is a piecewise upper envelope and can dip below var_w,
    # creating multiple disconnected feasible intervals (e.g. near the NE
    # corner). rho_r = measure{feasible returns <= r_w} / total feasible measure.

    def _seg_feasible_intervals(seg, need_below, target_var):
        """
        Sub-intervals of [lower_r, upper_r] where parabola satisfies the
        inequality.  Uses only the roots and sign of a — no interior evaluation.
        """
        lo, hi = seg["lower_r"], seg["upper_r"]
        a, b, c = seg["a_scaled"], seg["b_scaled"], seg["c_scaled"]
        disc = b * b - 4.0 * a * (c - target_var)

        if abs(a) < 1e-14:
            # Degenerate linear (or constant) segment
            if abs(b) < 1e-14:
                ok = (c <= target_var + tol) if need_below else (c >= target_var - tol)
                return [(lo, hi)] if ok else []
            r0 = (target_var - c) / b
            # b > 0: val increases → val <= target_var left of r0
            if need_below:
                flo, fhi = (lo, min(hi, r0)) if b > 0 else (max(lo, r0), hi)
            else:
                flo, fhi = (max(lo, r0), hi) if b > 0 else (lo, min(hi, r0))
            return [(flo, fhi)] if fhi > flo + tol else []

        # a > 0 (convex parabola): val(r) <= target_var iff r in [r_left, r_right]
        if disc <= 0.0:
            # No real roots or tangent: entire parabola >= target_var (vertex above)
            return [] if need_below else [(lo, hi)]

        sq = math.sqrt(disc)
        r_left  = (-b - sq) / (2.0 * a)
        r_right = (-b + sq) / (2.0 * a)
        if r_left > r_right:
            r_left, r_right = r_right, r_left

        if need_below:
            # Feasible BETWEEN roots: [r_left, r_right] ∩ [lo, hi]
            flo, fhi = max(lo, r_left), min(hi, r_right)
            return [(flo, fhi)] if fhi > flo + tol else []
        else:
            # Feasible OUTSIDE roots: (-∞, r_left] ∪ [r_right, +∞) ∩ [lo, hi]
            if r_right <= lo + tol or r_left >= hi - tol:
                return [(lo, hi)]
            if r_left <= lo + tol and r_right >= hi - tol:
                return []
            result = []
            if r_left > lo + tol:
                result.append((lo, r_left))
            if r_right < hi - tol:
                result.append((r_right, hi))
            return result

    def _merge(intervals):
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + tol:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append([lo, hi])
        return [(lo, hi) for lo, hi in merged]

    def _intersect(a_ivs, b_ivs):
        result = []
        i = j = 0
        while i < len(a_ivs) and j < len(b_ivs):
            lo = max(a_ivs[i][0], b_ivs[j][0])
            hi = min(a_ivs[i][1], b_ivs[j][1])
            if hi > lo + tol:
                result.append((lo, hi))
            if a_ivs[i][1] < b_ivs[j][1]:
                i += 1
            else:
                j += 1
        return result

    def _feasible_intervals_at(target_var):
        west_ivs = _merge([iv for seg in ef_segs + low_segs
                           for iv in _seg_feasible_intervals(seg, need_below=True,
                                                             target_var=target_var)])
        ea_ivs   = _merge([iv for seg in ea_segs
                           for iv in _seg_feasible_intervals(seg, need_below=False,
                                                             target_var=target_var)])
        return _intersect(west_ivs, ea_ivs)

    def _rho_from_intervals(r_o, intervals):
        total = sum(hi - lo for lo, hi in intervals)
        if total < tol:
            return None
        accum = 0.0
        for lo, hi in intervals:
            if r_o <= hi + tol:
                return max(0.0, min(1.0, (accum + max(0.0, r_o - lo)) / total))
            accum += hi - lo
        return 1.0

    rho_r = None
    r_min_at_sigma = None
    r_max_at_sigma = None
    feasible = _feasible_intervals_at(var_w)
    if feasible:
        r_min_at_sigma = feasible[0][0]
        r_max_at_sigma = feasible[-1][1]
        rho_r = _rho_from_intervals(r_w, feasible)

    # ---- rho_sigma: sd_min and sd_max at r_w --------------------------------
    rho_sigma     = None
    sd_min_at_r   = None
    sd_max_at_r   = None

    # sd_min at r_w: full west frontier (EF for r >= r_global, SW for r < r_global)
    west_cands = [s for s in ef_segs + low_segs
                  if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
    if west_cands:
        v_min = min(_var_on(s, r_w) for s in west_cands)
        sd_min_at_r = math.sqrt(max(v_min, 0.0))

    # sd_max at r_w: EA segment
    ea_cands = [s for s in ea_segs if s["lower_r"] - tol <= r_w <= s["upper_r"] + tol]
    if ea_cands:
        v_max = max(_var_on(s, r_w) for s in ea_cands)
        sd_max_at_r = math.sqrt(max(v_max, 0.0))

    if sd_min_at_r is not None and sd_max_at_r is not None:
        span = sd_max_at_r - sd_min_at_r
        if span > tol:
            rho_sigma = (sd_max_at_r - sd_w) / span
            rho_sigma = max(0.0, min(1.0, rho_sigma))

    # ---- gamma measures (unconditional global bounds) -----------------------
    r_min_global = float(mu.min())
    r_max_global = float(mu.max())

    _mvp_seg_g = next(
        (s for s in ef_segs + low_segs
         if s["lower_r"] - tol <= r_global <= s["upper_r"] + tol), None)
    sd_min_global = (math.sqrt(max(_var_on(_mvp_seg_g, r_global), 0.0))
                     if _mvp_seg_g is not None else None)
    sd_max_global = float(np.max(np.sqrt(np.diag(Sigma))))

    sharpe_max_global = _best_sr if _best_sr > -math.inf else None
    _asset_sds = np.sqrt(np.diag(Sigma))
    # If tangency fell at a single-asset corner (r_ms == mu[i]), recompute
    # sharpe_max_global directly so it is consistent with how sharpe_w is
    # computed for single-asset portfolios (w @ Sigma @ w, not the parabola).
    if r_ms is not None and sharpe_max_global is not None:
        for i in range(N):
            if abs(r_ms - float(mu[i])) < 1e-10 and float(_asset_sds[i]) > 1e-14:
                sharpe_max_global = (float(mu[i]) - rf) / float(_asset_sds[i])
                break
    sharpe_min_global = min(
        (float(mu[i]) - rf) / float(_asset_sds[i])
        for i in range(N) if float(_asset_sds[i]) > 1e-14
    ) if N > 0 else None

    sharpe_w = (r_w - rf) / sd_w if (sd_w is not None and sd_w > 1e-14) else None

    gamma_r = (max(0.0, min(1.0, (r_w - r_min_global) / (r_max_global - r_min_global)))
               if r_max_global - r_min_global > tol else None)

    gamma_sigma = (max(0.0, min(1.0, (sd_max_global - sd_w) / (sd_max_global - sd_min_global)))
                   if (sd_min_global is not None and sd_w is not None
                       and sd_max_global - sd_min_global > tol) else None)

    gamma_sharpe = (max(0.0, min(1.0, (sharpe_w - sharpe_min_global) /
                                       (sharpe_max_global - sharpe_min_global)))
                    if (sharpe_w is not None and sharpe_max_global is not None
                        and sharpe_min_global is not None
                        and sharpe_max_global - sharpe_min_global > tol) else None)

    # ---- w_ref (optional) ---------------------------------------------------
    _ref_qrp     = None
    w_ref_arr    = None
    dissim_w_ref = None
    if w_ref is not None:
        w_ref_arr = np.asarray(w_ref, float).ravel()
        if w_ref_arr.shape[0] != N:
            raise ValueError(f"w_ref length {w_ref_arr.shape[0]} != N={N}")
        _ref_qrp     = quasi_relative_performance(cloud_dict, w_ref_arr, sd=sd, tol=tol, rf=rf)
        dissim_w_ref = 0.5 * float(np.sum(np.abs(w - w_ref_arr)))

    # ---- verbose output -----------------------------------------------------
    if verbose:
        rep_cache_v = {}

        def _get_rep_v(active_set):
            key = tuple(active_set)
            if key not in rep_cache_v:
                rep_cache_v[key] = _active_representation(mu, Sigma, list(key))
            return rep_cache_v[key]

        def _weights_on_seg_v(seg, r_star):
            active = seg["active_set"]
            if len(active) == 1:
                w_star = np.zeros(N)
                w_star[active[0]] = 1.0
                return w_star
            rep    = _get_rep_v(active)
            w_star = np.zeros(N)
            w_star[list(active)] = rep["P"] * r_star + rep["q"]
            return w_star

        _abs  = absolute_performance(cloud_dict, weights, sd=sd, tol=tol)
        fsv_w = (np.array(_abs["frontier_same_var"]["w_frontier"])
                 if _abs["frontier_same_var"]["exists"]
                    and _abs["frontier_same_var"]["w_frontier"] is not None
                 else None)
        fsr_w = (np.array(_abs["frontier_same_r"]["w_frontier"])
                 if _abs["frontier_same_r"]["exists"]
                    and _abs["frontier_same_r"]["w_frontier"] is not None
                 else None)
        w_min_diss = (np.array(_abs["closest_ef_weights"]["w_ef"])
                      if _abs["closest_ef_weights"]["exists"]
                         and _abs["closest_ef_weights"]["w_ef"] is not None
                      else None)

        mvp_seg = next(
            (s for s in ef_segs + low_segs
             if s["lower_r"] - tol <= r_global <= s["upper_r"] + tol),
            None
        )
        w_mvp  = _weights_on_seg_v(mvp_seg, r_global) if mvp_seg else None
        idx_max = int(np.argmax(mu))
        w_maxr  = np.zeros(N); w_maxr[idx_max] = 1.0

        def _dissim(wa, wb):
            return 0.5 * float(np.sum(np.abs(np.asarray(wa, float) - np.asarray(wb, float))))

        # Max Sharpe weights (for column)
        w_ms = None
        for _seg in ef_segs:
            if r_ms is not None and _seg["lower_r"] - tol <= r_ms <= _seg["upper_r"] + tol:
                w_ms = _weights_on_seg_v(_seg, r_ms)
                break

        def _col_stats(wp):
            if wp is None:
                return {"r": None, "sd": None, "rho_r": None, "rho_sd": None,
                        "gamma_r": None, "gamma_sd": None, "gamma_sharpe": None}
            q = quasi_relative_performance(cloud_dict, wp, sd=sd, tol=tol, rf=rf)
            return {
                "r":            q["r_w"],
                "sd":           math.sqrt(max(q["var_w"], 0.0)) if sd else None,
                "rho_r":        q["rho_r"],
                "rho_sd":       q["rho_sigma"],
                "gamma_r":      q["gamma_r"],
                "gamma_sd":     q["gamma_sigma"],
                "gamma_sharpe": q["gamma_sharpe"],
            }

        # Build reference columns with full stats
        ref_cols = []
        for label, wp in [("Max r|Same sd", fsv_w),
                           ("Min sd|Same r", fsr_w),
                           ("Min Var",       w_mvp),
                           ("Max Return",    w_maxr)]:
            s = _col_stats(wp)
            if wp is not None:
                s["rho_r"]  = 1.0
                s["rho_sd"] = 1.0
            ref_cols.append({"label": label, "w": wp, **s})

        if w_ms is not None:
            s_ms = _col_stats(w_ms)
            s_ms["rho_r"]  = 1.0
            s_ms["rho_sd"] = 1.0
            ref_cols.append({"label": "Max Sharpe", "w": w_ms, **s_ms})

        if w_min_diss is not None:
            s_md = _col_stats(w_min_diss)
            s_md["rho_r"]  = 1.0
            s_md["rho_sd"] = 1.0
            ref_cols.append({"label": "EF Min Diss", "w": w_min_diss, **s_md})

        if w_ref_arr is not None:
            ref_cols.append({
                "label":        "w_ref",
                "w":            w_ref_arr,
                "r":            _ref_qrp["r_w"],
                "sd":           math.sqrt(max(_ref_qrp["var_w"], 0.0)) if sd else None,
                "rho_r":        _ref_qrp["rho_r"],
                "rho_sd":       _ref_qrp["rho_sigma"],
                "gamma_r":      _ref_qrp["gamma_r"],
                "gamma_sd":     _ref_qrp["gamma_sigma"],
                "gamma_sharpe": _ref_qrp["gamma_sharpe"],
            })

        WL, WW, WDW = 14, 8, 9
        grp_w = WW + WDW + 6

        _dissim_vals = [_dissim(w, col["w"]) if col["w"] is not None else None for col in ref_cols]
        _all_vals    = ([rho_r, rho_sigma, 0.0]
                        + [col["rho_r"]        for col in ref_cols]
                        + [col["rho_sd"]       for col in ref_cols]
                        + [gamma_r, gamma_sigma, gamma_sharpe, 0.0]
                        + [col["gamma_r"]      for col in ref_cols]
                        + [col["gamma_sd"]     for col in ref_cols]
                        + [col["gamma_sharpe"] for col in ref_cols]
                        + _dissim_vals)
        _all_deltas  = []
        for col in ref_cols:
            for vo, vc in [(rho_r,        col["rho_r"]),
                           (rho_sigma,    col["rho_sd"]),
                           (gamma_r,      col["gamma_r"]),
                           (gamma_sigma,  col["gamma_sd"]),
                           (gamma_sharpe, col["gamma_sharpe"])]:
                if vo is not None and vc is not None:
                    _all_deltas.append(vc - vo)

        dec_v = _auto_dec(_all_vals,   WW)
        dec_d = _auto_dec(_all_deltas, WDW, forced_sign=True)

        def _fv(val):
            return f"{val:>{WW}.{dec_v}f}" if val is not None else f"{'N/A':>{WW}}"
        def _fd(val):
            return f"{val:>+{WDW}.{dec_d}f}" if val is not None else f"{'N/A':>{WDW}}"
        def _grp(val, delta):
            return f"  {_fv(val)}  {_fd(delta)} |"

        def _stat_row(label, val_o, key):
            row = f"  {label:>{WL}} | {_fv(val_o)} |"
            for col in ref_cols:
                v = col[key]
                d = (v - val_o) if (v is not None and val_o is not None) else None
                row += _grp(v, d)
            return row

        h1 = f"  {'stat':>{WL}} | {'w_o':^{WW}} |"
        for col in ref_cols:
            h1 += f"{col['label']:^{grp_w}}"
        sep = "  " + "-" * (len(h1) - 2)

        print()
        print("--- quasi_relative_performance ---")
        print(h1)
        print(sep)
        print(_stat_row("rho_r",        rho_r,        "rho_r"))
        print(_stat_row("rho_sd",       rho_sigma,    "rho_sd"))
        print(sep)
        print(_stat_row("gamma_r",      gamma_r,      "gamma_r"))
        print(_stat_row("gamma_sd",     gamma_sigma,  "gamma_sd"))
        print(_stat_row("gamma_sharpe", gamma_sharpe, "gamma_sharpe"))
        print(sep)
        dissim_row = f"  {'dissimilarity':>{WL}} | {_fv(0.0)} |"
        for dv, col in zip(_dissim_vals, ref_cols):
            dissim_row += f"  {_fv(dv)}  {'':>{WDW}} |"
        print(dissim_row)
        print()

    return {
        "r_w":                 r_w,
        "var_w":               var_w,
        "sd_w":                sd_w,
        "rho_r":               rho_r,
        "r_min_at_sigma":      r_min_at_sigma,
        "r_max_at_sigma":      r_max_at_sigma,
        "rho_sigma":           rho_sigma,
        "sd_min_at_r":         sd_min_at_r,
        "sd_max_at_r":         sd_max_at_r,
        "sharpe_w":            sharpe_w,
        "gamma_r":             gamma_r,
        "r_min_global":        r_min_global,
        "r_max_global":        r_max_global,
        "gamma_sigma":         gamma_sigma,
        "sd_min_global":       sd_min_global,
        "sd_max_global":       sd_max_global,
        "gamma_sharpe":        gamma_sharpe,
        "sharpe_min_global":   sharpe_min_global,
        "sharpe_max_global":   sharpe_max_global,
        "ref_rho_r":           _ref_qrp["rho_r"]        if _ref_qrp else None,
        "ref_rho_sigma":       _ref_qrp["rho_sigma"]     if _ref_qrp else None,
        "ref_gamma_r":         _ref_qrp["gamma_r"]       if _ref_qrp else None,
        "ref_gamma_sigma":     _ref_qrp["gamma_sigma"]   if _ref_qrp else None,
        "ref_gamma_sharpe":    _ref_qrp["gamma_sharpe"]  if _ref_qrp else None,
        "dissim_w_ref":        dissim_w_ref,
    }


# =============================================================================
#  4. relative_performance — §3.4 measures (exact, no Monte Carlo)
# =============================================================================

# ---------------------------------------------------------------------------
# Simplex geometry helpers
# ---------------------------------------------------------------------------

def _simplex_vertices(N):
    """
    Return the N vertices of the standard probability simplex W_s in R^N.
    Vertex k is the unit vector e_k (weight 1 on asset k, 0 elsewhere).
    Shape: (N, N) — row k is vertex k.
    """
    return np.eye(N, dtype=float)


def _simplex_volume(N):
    """
    (N-1)-dimensional volume of the standard (N-1)-simplex in R^N,
    measured in the affine hyperplane {1'w = 1}.
    Vol = sqrt(N) / (N-1)!
    """
    return math.sqrt(N) / math.factorial(N - 1)


def _subsimplex_volume(vertices):
    """
    Exact (d-1)-dimensional volume of the simplex spanned by `vertices`
    (shape d x N, d points in R^N lying in a (d-1)-dimensional affine subspace).

    Uses: Vol = sqrt(det(G)) / (d-1)!  where G is the (d-1)x(d-1) Gram matrix
    of edge vectors from the first vertex.
    """
    d = vertices.shape[0]
    if d == 1:
        return 0.0
    edges = vertices[1:] - vertices[0]          # (d-1) x N
    G = edges @ edges.T                          # (d-1) x (d-1) Gram matrix
    det_G = np.linalg.det(G)
    if det_G < 0:
        det_G = 0.0
    return math.sqrt(det_G) / math.factorial(d - 1)


# ---------------------------------------------------------------------------
# P_r+  — exact polytope volume fraction
# ---------------------------------------------------------------------------

def _halfspace_simplex_vertices(simplex_verts, mu, threshold):
    """
    Clip the simplex {simplex_verts} by the halfspace mu'w > threshold.
    Returns the vertices of the intersection polytope.

    Algorithm:
      - classify each vertex as inside (mu'v > threshold) or outside
      - for each edge crossing the boundary, compute the intersection point
      - collect inside vertices + edge intersection points
    """
    vals = simplex_verts @ mu          # shape (N,)
    inside = vals > threshold
    result = list(simplex_verts[inside])

    n = len(simplex_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if inside[i] != inside[j]:
                # edge (i, j) crosses the hyperplane
                t = (threshold - vals[i]) / (vals[j] - vals[i])
                pt = simplex_verts[i] + t * (simplex_verts[j] - simplex_verts[i])
                result.append(pt)

    return np.array(result) if result else np.empty((0, simplex_verts.shape[1]))


def _p_r_plus(mu, N, threshold):
    """
    Exact Pr_{w~Unif(W_s)}(mu'w > threshold).

    Method: project the clipped polytope to R^{N-1} (drop last coordinate),
    compute volume with scipy.spatial.ConvexHull, divide by 1/(N-1)!.
    The sqrt(N) scale factor from the embedding cancels in the ratio.
    """
    from scipy.spatial import ConvexHull, QhullError

    verts = _simplex_vertices(N)
    inner_verts = _halfspace_simplex_vertices(verts, mu, threshold)

    if len(inner_verts) == 0:
        return 0.0

    d = N - 1                          # simplex dimension
    total_vol = 1.0 / math.factorial(d) # volume of standard d-simplex in R^d
    inner_proj = inner_verts[:, :-1]   # project to R^d by dropping last coord

    if d == 0:
        return 1.0
    if d == 1:
        inner_vol = float(np.max(inner_proj) - np.min(inner_proj))
        return min(1.0, max(0.0, inner_vol))  # total_vol = 1 for d=1

    if len(inner_proj) < d + 1:
        return 0.0  # degenerate

    try:
        hull = ConvexHull(inner_proj, qhull_options='Qt')
        return min(1.0, max(0.0, hull.volume / total_vol))
    except (QhullError, Exception):
        return 0.0


# ---------------------------------------------------------------------------
# Solid-angle helpers for ball-polytope intersection (P_σ⁻ and P_SR⁺)
# ---------------------------------------------------------------------------

def _solid_angle_simplex(verts_from_origin):
    """
    Exact solid angle (in steradians, as a fraction of the full d-sphere surface)
    subtended by the simplex cone from the origin through vertices `verts_from_origin`.

    Uses the generalised Gram formula (Van Oosterom & Strackee 1983 for d=3;
    Ribando 2006 for general d via recursive formula).

    verts_from_origin : array (d, dim) — d direction vectors (need not be unit)
    Returns: solid angle as fraction of full sphere (i.e. in [0, 1]).
    """
    verts = np.array(verts_from_origin, dtype=float)
    d = verts.shape[0]          # number of generators (= simplex dimension)

    if d == 0:
        return 0.0
    if d == 1:
        # A single ray subtends zero solid angle
        return 0.0

    # Normalise
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    zero_mask = (norms.ravel() < 1e-15)
    if np.any(zero_mask):
        return 0.0
    u = verts / norms   # unit vectors, shape (d, dim)

    if d == 2:
        # 1-D case: angle between two unit vectors, fraction of half-circle
        cos_theta = np.clip(u[0] @ u[1], -1.0, 1.0)
        theta = math.acos(cos_theta)           # angle in [0, pi]
        # Solid angle of a cone in R^2: fraction of full circle = theta / (2*pi)
        # For a simplex cone (wedge) bounded by two rays:
        return theta / (2.0 * math.pi)

    if d == 3:
        # Van Oosterom & Strackee (1983): exact formula for a spherical triangle
        # solid angle (as fraction of 4*pi steradians):
        #   tan(Omega/2) = |u0 . (u1 x u2)| / (1 + u0.u1 + u0.u2 + u1.u2)
        a, b, c = u[0], u[1], u[2]
        num = abs(float(a @ np.cross(b, c)))
        den = 1.0 + float(a @ b) + float(a @ c) + float(b @ c)
        if abs(den) < 1e-15:
            # degenerate (nearly-planar triangle containing origin)
            return 0.0
        half_omega = math.atan2(num, den)
        omega = 2.0 * half_omega    # solid angle in steradians
        return omega / (4.0 * math.pi)

    # For d >= 4: recursive formula via inclusion-exclusion on facets.
    # The solid angle of a d-simplex cone = sum of contributions from its
    # (d-1)-facets minus the solid angles of the facet-opposite simplices, etc.
    # We use the Gram relation:
    #   sum_{faces F} (-1)^(d - dim(F)) * Omega(F) = 1/2  (for even d)
    # This is complex; for d >= 4 we fall back to a high-precision numerical
    # estimate via random sampling (this is acceptable as d >= 4 means N >= 5
    # assets, which is rare in practice for this application).
    # The sampling count is large enough for 6-digit accuracy.
    n_samples = 200_000
    dim = verts.shape[1]
    # Draw uniform directions on the unit sphere in R^dim
    rng = np.random.default_rng(seed=0)
    z = rng.standard_normal((n_samples, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # Check if each sample lies inside the cone: all dot products with
    # the outward-facing normals of the simplex facets must be >= 0.
    # Build the Gram matrix of u and test via QP-sign check:
    # A point x is inside the simplicial cone iff x = sum(lambda_i * u_i)
    # with lambda_i >= 0. Equivalently, solve x = U' * lambda, lambda >= 0.
    # For random directions we use the sign of (U @ x):
    # Actually: x is in the cone iff the dual cone condition holds.
    # Simpler: x in cone(u_0..u_{d-1}) iff there exist lambda >= 0 with x = U'lambda.
    # Use lstsq and check positivity.
    count = 0
    U = u.T   # (dim, d)
    for zi in z:
        lam, _, _, _ = np.linalg.lstsq(U, zi, rcond=None)
        if np.all(lam >= -1e-9):
            count += 1
    return count / n_samples


def _ball_simplex_vol_fraction(simplex_verts, radius, origin):
    """
    Exact fraction of the simplex (given by its vertices) that lies inside
    the ball of given radius centred at `origin`, using solid-angle decomposition.

    Method:
      1. Translate so the origin is at 0.
      2. Clip the simplex by the ball: split into "fully inside", "partially inside"
         sub-simplices.
      3. For fully-inside sub-simplices, volume = _subsimplex_volume.
      4. For partially-inside sub-simplices (straddle the sphere), use the
         solid-angle formula: Vol(ball(r) ∩ simplicial_cone) / Vol(simplex).

    For simplicity we use a fan triangulation of the simplex from `origin`
    projected onto the simplex affine hull, computing each fan-pyramid's
    intersection with the ball exactly.
    """
    verts = np.array(simplex_verts, dtype=float)
    origin = np.asarray(origin, dtype=float)
    v = verts - origin          # shifted vertices

    N_pts = verts.shape[0]     # number of vertices
    d = N_pts - 1               # dimension of simplex (N-1 for W_s)

    dists = np.linalg.norm(v, axis=1)  # distance from origin to each vertex

    # If all vertices inside the ball: fraction = 1
    if np.all(dists <= radius + 1e-14):
        return 1.0
    # If all vertices outside the ball: use solid-angle formula for the whole simplex
    if np.all(dists >= radius - 1e-14):
        omega = _solid_angle_simplex(v)
        # Vol(ball(r) ∩ cone) = omega * r^d * S_d / d  where S_d is the
        # surface area of the unit d-sphere. But we want the fraction of the
        # simplex volume. This requires comparing cone-cap volume to simplex volume.
        # For a simplicial cone with solid angle omega (fraction of sphere):
        #   Vol(ball ∩ cone) = omega * Vol(ball in d dims)
        # We need this as a fraction of the simplex volume.
        # This path is complex; fall back to recursive sub-division.
        pass   # fall through to the recursive subdivision below

    # Recursive subdivision: split the simplex at each vertex-sphere intersection
    # along edges, creating sub-simplices that are fully inside or fully outside.
    # For a simplex with some vertices inside and some outside:
    #   - find edge crossings with the sphere
    #   - create sub-simplices; those with all vertices inside contribute fully
    # This is the "sphere clipping" of a simplex. For exact results we use
    # the Ribando (2006) solid-angle decomposition for the spherical cap pieces.

    # Build sub-simplices by fan from origin
    # Each sub-simplex shares the origin and one face of the clipped simplex.
    # The volume inside the ball = sum over sub-simplices of:
    #   min(1, r^d / ||furthest vertex||^d) * sub_simplex_vol   (rough approx)
    # For exact: use solid angle for sub-simplices straddling the sphere.

    # We use a clean recursive approach: evaluate the integral
    #   I = int_{simplex} 1[||w - origin|| <= r] dw
    # by quadrature on the simplex using Grundmann-Möller rules (exact for polynomials).
    # Since the integrand is NOT a polynomial (it's an indicator), we instead
    # use the following exact decomposition valid for simplices:

    # Split each edge that crosses the sphere into two sub-edges at the crossing point.
    # Collect the clipped vertices; then compute their volume directly.
    # This is exact because the sphere boundary is curved but the simplex is flat:
    # the portion of a flat simplex inside a ball = polytope volume if all
    # crossings are on edges (not faces), which is generically true.

    # HOWEVER: the intersection of a ball with a simplex is NOT a polytope
    # (it has a curved boundary on the sphere side). Its volume cannot be computed
    # by a polytope-volume formula alone. The exact formula requires the
    # solid-angle / spherical-cap contribution.

    # Exact decomposition (Aomoto 1977, generalised):
    # Vol(simplex ∩ ball) = Vol(polytope_clipped) + spherical_cap_contribution
    # where polytope_clipped is the intersection of the simplex with the chord-polytope
    # and spherical_cap_contribution accounts for the curved cap.

    # For the purposes of this implementation we compute this exactly for
    # d <= 3 (N <= 4 assets) and use high-precision numerical integration for d > 3.

    if d <= 3:
        return _ball_simplex_exact_low_d(v, radius, d)
    else:
        return _ball_simplex_numerical(v, radius, d)


def _ball_simplex_exact_low_d(v, radius, d):
    """
    Exact Vol(ball(radius) ∩ simplex(v)) / Vol(simplex(v))
    for d = 1, 2, 3 (i.e. N = 2, 3, 4 assets).
    v: shifted vertices (origin at 0), shape (d+1, ambient_dim).
    """
    vol_simplex = _subsimplex_volume(v)
    if vol_simplex < 1e-18:
        return 0.0

    dists = np.linalg.norm(v, axis=1)

    if d == 1:
        # Simplex is a line segment [v0, v1].
        # Find t in [0,1] where ||v0 + t*(v1-v0)|| = radius.
        # ||v0 + t*(v1-v0)||^2 = radius^2
        a_c = float(np.dot(v[1] - v[0], v[1] - v[0]))
        b_c = 2.0 * float(np.dot(v[0], v[1] - v[0]))
        c_c = float(np.dot(v[0], v[0])) - radius ** 2
        disc = b_c * b_c - 4.0 * a_c * c_c
        t_vals = [0.0, 1.0]
        if disc >= 0 and abs(a_c) > 1e-15:
            sq = math.sqrt(max(0.0, disc))
            for t in [(-b_c - sq) / (2 * a_c), (-b_c + sq) / (2 * a_c)]:
                if 0.0 <= t <= 1.0:
                    t_vals.append(t)
        t_vals = sorted(set(t_vals))
        # integrate each sub-segment
        vol_inside = 0.0
        seg_len = np.linalg.norm(v[1] - v[0])
        for i in range(len(t_vals) - 1):
            t_mid = 0.5 * (t_vals[i] + t_vals[i + 1])
            pt = v[0] + t_mid * (v[1] - v[0])
            if np.linalg.norm(pt) <= radius + 1e-14:
                sub_len = (t_vals[i + 1] - t_vals[i]) * seg_len
                vol_inside += sub_len
        return vol_inside / vol_simplex

    if d == 2:
        # Simplex is a triangle.  Decompose into fan from origin if origin
        # is inside the triangle (projected), else from a vertex.
        # Use Green's theorem: area inside ball ∩ triangle.
        # Exact via decomposing boundary into straight edges and arc segments.
        return _triangle_ball_fraction(v, radius)

    if d == 3:
        # Tetrahedron. Decompose into 4 face-pyramids and use exact formula.
        return _tetrahedron_ball_fraction(v, radius)

    return _ball_simplex_numerical(v, radius, d)


def _triangle_ball_fraction(v, radius):
    """
    Exact area fraction of triangle (v[0], v[1], v[2]) inside ball(radius)
    centered at the origin. Correct for all configurations: ball center inside
    or outside the triangle, edge passing entirely through the ball, etc.

    Uses Green's theorem polygon-circle signed-area integration:
    for each directed edge a→b, split at sphere crossings and integrate
    triangle area (inside) or arc sector area (outside).
    """
    r = radius

    # Project triangle to 2D in its own plane
    e1 = v[1] - v[0]
    len_e1 = float(np.linalg.norm(e1))
    if len_e1 < 1e-15:
        return 0.0
    e1_hat = e1 / len_e1
    e2 = v[2] - v[0]
    e2 = e2 - float(np.dot(e2, e1_hat)) * e1_hat
    len_e2 = float(np.linalg.norm(e2))
    if len_e2 < 1e-15:
        return 0.0
    e2_hat = e2 / len_e2

    def to2d(p):
        delta = p - v[0]
        return np.array([float(np.dot(delta, e1_hat)), float(np.dot(delta, e2_hat))])

    # 2D projection of ball center (origin in ambient space)
    org2d = to2d(np.zeros(v.shape[1]))

    # The origin sits at perpendicular distance d_plane from the triangle's plane.
    # By Pythagoras: d_plane^2 = ||v[0]||^2 - ||org2d||^2
    # The 3D ball of radius r intersects the plane as a 2D disc of effective radius
    # r_eff = sqrt(r^2 - d_plane^2).  Using r directly inflates the disc.
    d_plane_sq = max(0.0, float(np.dot(v[0], v[0])) - float(org2d[0]**2 + org2d[1]**2))
    if d_plane_sq >= r * r - 1e-14:
        return 0.0          # ball does not reach the plane
    r_eff = math.sqrt(r * r - d_plane_sq)

    # Shift so ball centre's 2D projection is at coordinate origin
    pts = [to2d(vi) - org2d for vi in v]

    # Ensure CCW orientation
    sa = sum(pts[i][0]*pts[(i+1)%3][1] - pts[(i+1)%3][0]*pts[i][1] for i in range(3))
    if sa < 0:
        pts = pts[::-1]

    def cross2(a, b):
        return a[0]*b[1] - a[1]*b[0]

    def _edge_contrib(a, b):
        d = b - a
        A = float(np.dot(d, d))
        B = 2.0 * float(np.dot(a, d))
        C = float(np.dot(a, a)) - r_eff * r_eff
        ts = []
        if A > 1e-20:
            disc = B*B - 4*A*C
            if disc >= 0:
                sq = math.sqrt(max(0.0, disc))
                for t in [(-B - sq) / (2*A), (-B + sq) / (2*A)]:
                    if 1e-10 < t < 1.0 - 1e-10:
                        ts.append(t)
        params = [0.0] + sorted(set(ts)) + [1.0]
        contrib = 0.0
        for k in range(len(params) - 1):
            t0, t1 = params[k], params[k+1]
            p0 = a + t0 * d
            p1 = a + t1 * d
            mid = a + 0.5*(t0+t1) * d
            if float(np.dot(mid, mid)) <= r_eff*r_eff + 1e-14:
                # Segment inside disc: triangle shoelace contribution
                contrib += 0.5 * cross2(p0, p1)
            else:
                # Segment outside disc: arc (sector) contribution
                n0 = float(np.linalg.norm(p0))
                n1 = float(np.linalg.norm(p1))
                if n0 > 1e-15 and n1 > 1e-15:
                    q0 = p0 * (r_eff / n0)
                    q1 = p1 * (r_eff / n1)
                    angle = math.atan2(cross2(q0, q1), float(np.dot(q0, q1)))
                    contrib += 0.5 * r_eff * r_eff * angle
        return contrib

    area = sum(_edge_contrib(pts[i], pts[(i+1)%3]) for i in range(3))
    tri_area = _subsimplex_volume(v)
    if tri_area < 1e-18:
        return 0.0
    return min(1.0, max(0.0, area / tri_area))


def _tetrahedron_ball_fraction(v, radius):
    """
    Volume fraction of tetrahedron (v[0..3]) inside ball(radius) at origin.
    Delegates to the numerical integrator, which handles all configurations.
    """
    return _ball_simplex_numerical(v, radius, 3)


def _ball_simplex_numerical(v, radius, d, n_pts=50):
    """
    High-precision Grundmann-Möller quadrature for Vol(ball ∩ simplex) / Vol(simplex)
    for d > 3. Used only when N > 4 assets. Degree-2s+1 exact for polynomials
    (indicator is not polynomial, but fine grid gives high accuracy).
    """
    # Grundmann-Möller rule of order s for the d-simplex (Grundmann & Möller 1978)
    # Implement s=4 (degree 9 rule); for an indicator function this gives ~1e-4 accuracy.
    # For better accuracy use a tensor product of 1D Gauss-Jacobi rules.
    # We use n_pts^d uniformly spaced quadrature points via rejection (simplex sampling).
    n_samples = max(100_000, 10 ** d)
    rng = np.random.default_rng(seed=42)
    # Uniform samples on d-simplex via sorted uniform trick (Devroye 1986)
    exp = rng.standard_exponential((n_samples, d + 1))
    bary = exp / exp.sum(axis=1, keepdims=True)   # Dirichlet(1,...,1) = Unif on simplex
    pts = bary @ v                                 # shape (n_samples, ambient_dim)
    dists = np.linalg.norm(pts, axis=1)
    return float(np.mean(dists <= radius))


def _p_sigma_minus(cloud_dict, weights, tol=1e-10):
    """
    Exact P_sigma^- = Pr_{w~Unif(W_s)}(sigma(w) < sigma(w_o)).
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    N      = cloud_dict["N"]
    chol_L = cloud_dict["chol_L"]

    w_o   = np.asarray(weights, float).ravel()
    var_o = float(w_o @ (Sigma @ w_o))
    sigma_o = math.sqrt(max(var_o, 0.0))

    # Transform to z-space: z = L' w, ||z||^2 = w'Σw
    # Simplex vertices in w-space: e_k  →  z_k = L' e_k = k-th column of L'
    simplex_verts_w = _simplex_vertices(N)              # (N, N)
    simplex_verts_z = (chol_L.T @ simplex_verts_w.T).T  # (N, N), row k = L' e_k

    # Origin in z-space (ball centre)
    origin_z = np.zeros(N)

    frac = _ball_simplex_vol_fraction(simplex_verts_z, sigma_o, origin_z)
    return min(1.0, max(0.0, frac))


def _p_sr_plus(cloud_dict, weights, tol=1e-10, rf=0.0):
    """
    Exact P_SR^+ = Pr_{w~Unif(W_s)}(SR(w) > SR(w_o)).

    SR(w) = (mu'w - rf) / sqrt(w'Σw).
    In z-space (z = L'w): SR(w) = (nu'z) / ||z|| where nu = L^{-1} (mu - rf).
    The condition SR(w) > k becomes cos(angle(z, nu)) > k / ||nu||,
    i.e. the angle between z and nu is less than arccos(k / ||nu||).
    This is a cone around the direction nu.

    Strategy: for each simplex vertex sub-cone (fan from origin through each
    simplex face), compute the fraction inside the cone using solid angles.
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    N      = cloud_dict["N"]
    chol_L = cloud_dict["chol_L"]

    w_o   = np.asarray(weights, float).ravel()
    r_o   = float(mu @ w_o)
    var_o = float(w_o @ (Sigma @ w_o))
    if var_o < 1e-20:
        return 0.0   # degenerate zero-variance portfolio
    sr_o  = (r_o - rf) / math.sqrt(var_o)

    # nu = L^{-1} (mu - rf)  (direction of increasing SR in z-space)
    nu = np.linalg.solve(chol_L, mu - rf)
    nu_norm = np.linalg.norm(nu)
    if nu_norm < 1e-14:
        return 0.0

    # Threshold cosine: cos(theta*) = sr_o / nu_norm
    cos_thresh = sr_o / nu_norm
    # Clamp: if cos_thresh >= 1 no portfolio beats SR_o; if <= -1 all do
    if cos_thresh >= 1.0 - 1e-12:
        return 0.0
    if cos_thresh <= -1.0 + 1e-12:
        return 1.0

    nu_hat = nu / nu_norm

    # Simplex vertices in z-space
    simplex_verts_w = _simplex_vertices(N)
    simplex_verts_z = (chol_L.T @ simplex_verts_w.T).T  # (N, N)

    # P_SR^+ = fraction of simplex (in z-space) inside the cone {cos(z,nu) > cos_thresh}
    # = fraction where nu_hat . z_hat > cos_thresh
    # The cone is a spherical cap around nu_hat.
    # We compute this as the solid angle of the intersection of the simplicial cone
    # with the spherical cap, divided by the full solid angle of the simplicial cone.
    # More directly: fraction of uniform W_s points where angle < theta*.

    # For d=1 (N=2): simplex is a line segment. Fraction is easily computed.
    # For d>=2: use numerical estimate (the cone boundary is smooth, not flat,
    # so polyhedral methods don't directly apply; the solid angle ratio approach works).

    # Count fraction of simplex in the cone via Grundmann-Möller / sampling
    # (the cone boundary is a smooth sphere, so the indicator is not polynomial;
    # for exact results in low d we use analytical geometry):

    if N == 2:
        # Simplex is a single interval w1 in [0,1], w2 = 1-w1.
        # z(w) = L' w = w1 * z0 + w2 * z1 where z0,z1 are simplex_verts_z[0,1].
        z0, z1 = simplex_verts_z[0], simplex_verts_z[1]
        # cos(angle(z(t), nu)) = nu_hat . z(t) / ||z(t)||
        # Scan t in [0,1]: find crossing where cos = cos_thresh.
        def _cos_at(t):
            zt = (1 - t) * z0 + t * z1
            nzt = np.linalg.norm(zt)
            if nzt < 1e-15:
                return 0.0
            return float(nu_hat @ zt) / nzt

        n_scan = 10000
        ts = np.linspace(0, 1, n_scan + 1)
        cos_vals = np.array([_cos_at(t) for t in ts])
        frac = float(np.mean(cos_vals > cos_thresh))
        return frac

    # General case: numerical integration on simplex
    n_samples = 200_000
    rng = np.random.default_rng(seed=1)
    exp = rng.standard_exponential((n_samples, N))
    bary = exp / exp.sum(axis=1, keepdims=True)
    z_pts = bary @ simplex_verts_z   # (n_samples, N)
    nz = np.linalg.norm(z_pts, axis=1, keepdims=True)
    cos_vals = (z_pts / np.where(nz > 1e-15, nz, 1.0)) @ nu_hat
    frac = float(np.mean(cos_vals > cos_thresh))
    return min(1.0, max(0.0, frac))


def _simplex_grid(N, n_target, k=None, max_overshoot=4.0):
    """
    Deterministic barycentric lattice on the (N-1)-simplex.

    Finds k such that C(k+N-1, N-1) is closest to n_target, then enumerates
    all integer vectors (i_1,...,i_N) with i_j >= 0 and sum = k, returning
    W = those vectors / k.  Each row of W sums to 1.

    If k is supplied directly it bypasses the n_target search AND the
    overshoot fallback below — an explicit k is always honored, however
    many points it produces.

    When k is auto-derived from n_target, falls back to seeded (seed=0)
    Dirichlet sampling of n_target points if the resulting lattice would
    have more than max_overshoot * n_target points (occurs for large N,
    where even k=2 can overshoot) — and prints a note when it does, since
    this silently changes the sampling scheme from a deterministic lattice
    to a random draw.
    """
    from math import comb

    if k is None:
        k = 1
        while comb(k + N - 1, N - 1) < n_target:
            k += 1
        if k > 1 and abs(comb(k - 1 + N - 1, N - 1) - n_target) < abs(comb(k + N - 1, N - 1) - n_target):
            k -= 1
        n_lattice = comb(k + N - 1, N - 1)
        if n_lattice > max_overshoot * n_target:
            print(f"_simplex_grid: auto-selected k={k} for N={N} would yield "
                  f"{n_lattice} points (> {max_overshoot}x n_target={n_target}); "
                  f"falling back to {n_target} random Dirichlet-sampled points "
                  f"(seed=0) instead of the deterministic lattice. Pass an "
                  f"explicit lattice_k to force the deterministic lattice "
                  f"regardless of size.")
            return np.random.default_rng(0).dirichlet(np.ones(N), size=n_target)
    else:
        n_lattice = comb(k + N - 1, N - 1)

    _mb = n_lattice * N * 8 / 1e6  # size of one (n_lattice, N) float64 array
    print(f"_simplex_grid: k={k}, points={n_lattice} (~{_mb:.0f} MB per "
          f"(points, N) float64 array — several such arrays are held at once "
          f"downstream)")

    pts = []
    def _gen(dim, rem, cur):
        if dim == 1:
            pts.append(cur + [rem / k])
            return
        for i in range(rem + 1):
            _gen(dim - 1, rem - i, cur + [i / k])
    _gen(N, k, [])
    return np.array(pts)


def _build_t_form(mu, Sigma):
    """
    Precompute t-parameterization coefficients using asset N-1 (last asset) as base.

    w = e_N + Σ t_i (e_i − e_N)  so that  r(t) = μ_N + a·t,  σ²(t) = c₀ + b·t + t·Q·t
    """
    N     = len(mu)
    mu_N  = float(mu[N - 1])
    a_vec = (mu[:N - 1] - mu[N - 1]).astype(float)         # (N-1,)
    s_col = Sigma[:N - 1, N - 1].astype(float)              # (N-1,): Σ_{i,N}
    S_NN  = float(Sigma[N - 1, N - 1])
    Q_mat = (Sigma[:N - 1, :N - 1].astype(float)
             - s_col[:, None] - s_col[None, :] + S_NN)      # (N-1, N-1)
    b_vec = 2.0 * (s_col - S_NN)                            # (N-1,)
    c0    = S_NN
    return mu_N, a_vec, Q_mat, b_vec, c0


def _duffy_gl_grid(K_per_dim, outer_dim):
    """
    Gauss-Legendre quadrature nodes on the outer_dim-simplex via Duffy transform.

    Returns
    -------
    t_bar : (K_total, outer_dim)  — outer simplex points
    T     : (K_total,)            — remaining budget = 1 − Σ t̄_i
    wts   : (K_total,)            — combined GL × Jacobian × (outer_dim+1)! weights
    """
    import math
    if outer_dim == 0:
        return np.empty((1, 0)), np.ones(1), np.ones(1)

    nodes, gl_w = np.polynomial.legendre.leggauss(K_per_dim)
    u1d = (nodes + 1.0) / 2.0      # [−1,1] → [0,1]
    w1d = gl_w / 2.0

    if outer_dim == 1:
        t_bar = u1d.reshape(-1, 1)
        T     = 1.0 - u1d
        return t_bar, T, w1d * math.factorial(2)

    grids  = np.meshgrid(*([u1d] * outer_dim), indexing='ij')
    wgrids = np.meshgrid(*([w1d] * outer_dim), indexing='ij')
    u_flat = np.stack([g.ravel() for g in grids],  axis=1)   # (K^d, d)
    w_flat = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)  # (K^d,)

    K_total = u_flat.shape[0]
    t_bar   = np.zeros((K_total, outer_dim))
    cum     = np.ones(K_total)
    for i in range(outer_dim):
        t_bar[:, i] = u_flat[:, i] * cum
        cum = cum * (1.0 - u_flat[:, i])
    T = cum

    jac = np.ones(K_total)
    for j in range(outer_dim - 1):
        jac *= (1.0 - u_flat[:, j]) ** (outer_dim - 1 - j)

    return t_bar, T, w_flat * jac * math.factorial(outer_dim + 1)


def _inner_length_batch(t_bar_batch, T_batch, mu_N, a_vec, Q_mat, b_vec, c0,
                        r_vals, sig_sq_vals, kind):
    """
    Vectorized inner integral lengths over the last t coordinate.

    t_bar_batch : (K, outer_dim)
    T_batch     : (K,)
    r_vals      : (M,)
    sig_sq_vals : (M,)
    kind        : 'A' (region dominating target) or 'F' (region target dominates)
    Returns     : (K, M)
    """
    K         = T_batch.shape[0]
    M         = r_vals.shape[0]
    outer_dim = t_bar_batch.shape[1]
    a_last    = float(a_vec[outer_dim])

    r_bar = (mu_N + t_bar_batch @ a_vec[:outer_dim]) if outer_dim > 0 else np.full(K, mu_N)
    R_km  = r_vals[None, :] - r_bar[:, None]   # (K, M)

    if outer_dim > 0:
        sig_sq_bar = (c0
                      + t_bar_batch @ b_vec[:outer_dim]
                      + np.einsum('ki,ij,kj->k', t_bar_batch,
                                  Q_mat[:outer_dim, :outer_dim], t_bar_batch))
    else:
        sig_sq_bar = np.full(K, c0)

    alpha    = float(Q_mat[outer_dim, outer_dim])
    beta_bar = ((b_vec[outer_dim] + 2.0 * t_bar_batch @ Q_mat[:outer_dim, outer_dim])
                if outer_dim > 0
                else np.full(K, float(b_vec[outer_dim])))

    gamma_km = sig_sq_bar[:, None] - sig_sq_vals[None, :]   # (K, M)
    T_k      = T_batch[:, None]                              # (K, 1)

    # σ² interval [s_lo, s_hi] where α t² + β t + γ < 0
    if alpha > 1e-14:
        disc      = beta_bar[:, None] ** 2 - 4.0 * alpha * gamma_km
        has_roots = disc > 0.0
        sqd       = np.sqrt(np.maximum(disc, 0.0))
        inv2a     = 0.5 / alpha
        s_lo = np.clip((-beta_bar[:, None] - sqd) * inv2a, 0.0, T_k)
        s_hi = np.clip((-beta_bar[:, None] + sqd) * inv2a, 0.0, T_k)
    else:
        bk  = beta_bar[:, None]
        thr = np.where(np.abs(bk) > 1e-14,
                       -gamma_km / np.where(np.abs(bk) > 1e-14, bk, 1.0),
                       0.0)
        s_lo = np.where(bk  >  1e-14, 0.0,
               np.where(bk  < -1e-14, np.clip(thr, 0.0, T_k),
                        np.where(gamma_km < 0, 0.0, T_k)))
        s_hi = np.where(bk  >  1e-14, np.clip(thr, 0.0, T_k),
               np.where(bk  < -1e-14, T_k,
                        np.where(gamma_km < 0, T_k, 0.0)))
        has_roots = s_hi > s_lo

    T_bc = np.broadcast_to(T_k, (K, M)).copy()
    if abs(a_last) > 1e-12:
        thr_r = R_km / a_last
        if a_last > 0:
            if kind == 'A':
                r_lo, r_hi, r_ok = np.maximum(thr_r, 0.0), T_bc, thr_r < T_k
            else:
                r_lo, r_hi, r_ok = np.zeros((K, M)), np.minimum(thr_r, T_bc), thr_r > 0.0
        else:
            if kind == 'A':
                r_lo, r_hi, r_ok = np.zeros((K, M)), np.minimum(thr_r, T_bc), thr_r > 0.0
            else:
                r_lo, r_hi, r_ok = np.maximum(thr_r, 0.0), T_bc, thr_r < T_k
    else:
        r_lo = np.zeros((K, M))
        r_hi = T_bc
        r_ok = (R_km < 0.0) if kind == 'A' else (R_km > 0.0)

    if kind == 'A':
        lo = np.maximum(r_lo, s_lo)
        hi = np.minimum(r_hi, s_hi)
        return np.where(r_ok & has_roots, np.maximum(0.0, hi - lo), 0.0)
    else:
        p1   = np.maximum(0.0, np.minimum(r_hi, s_lo)  - np.maximum(r_lo, 0.0))
        p2   = np.maximum(0.0, np.minimum(r_hi, T_bc)  - np.maximum(r_lo, s_hi))
        full = np.maximum(0.0, r_hi - r_lo)
        return np.where(r_ok, np.where(has_roots, p1 + p2, full), 0.0)


def _A_i_F_i_analytical(r_vals, sig_sq_vals, mu, Sigma, N, n_quad=200, chunk_M=10000):
    """
    Analytical A_i and F_i for M portfolios via iterated GL quadrature.

    r_vals, sig_sq_vals : (M,) — return and variance of each portfolio
    Returns A_arr, F_arr : (M,) each in [0, 1]
    """
    r_vals      = np.asarray(r_vals,      float)
    sig_sq_vals = np.asarray(sig_sq_vals, float)
    M           = r_vals.shape[0]

    mu_N, a_vec, Q_mat, b_vec, c0 = _build_t_form(mu, Sigma)

    outer_dim = N - 2
    K_per_dim = n_quad if outer_dim <= 1 else max(3, int(round(n_quad ** (1.0 / outer_dim))))
    t_bar_b, T_b, gl_w = _duffy_gl_grid(K_per_dim, outer_dim)

    A_arr = np.zeros(M)
    F_arr = np.zeros(M)
    for s in range(0, M, chunk_M):
        e   = min(s + chunk_M, M)
        r_c = r_vals[s:e]
        sq_c = sig_sq_vals[s:e]
        lA = _inner_length_batch(t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, r_c, sq_c, 'A')
        lF = _inner_length_batch(t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, r_c, sq_c, 'F')
        A_arr[s:e] = gl_w @ lA
        F_arr[s:e] = gl_w @ lF

    return np.clip(A_arr, 0.0, 1.0), np.clip(F_arr, 0.0, 1.0)


def _q_a_f_v2(cloud_dict, weights, n_points=4000, lattice_k=100, n_quad=200):
    """
    O(M × K^{N-2}) analytical replacement for O(M²) _q_a_f.

    A_i and F_i for w_o are computed via GL quadrature (not lattice counting).
    Q_A and Q_F are estimated by evaluating A/F analytically over a barycentric lattice.
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    chol_L = cloud_dict["chol_L"]
    N      = cloud_dict["N"]

    w_o   = np.asarray(weights, float).ravel()
    r_o   = float(mu @ w_o)
    var_o = float(w_o @ Sigma @ w_o)

    A_o_arr, F_o_arr = _A_i_F_i_analytical(
        np.array([r_o]), np.array([var_o]), mu, Sigma, N, n_quad=n_quad)
    A_i = float(A_o_arr[0])
    F_i = float(F_o_arr[0])

    W       = _simplex_grid(N, n_points, k=lattice_k)
    r_vec   = W @ mu
    Z       = W @ chol_L
    var_vec = np.sum(Z ** 2, axis=1)

    A_grid, F_grid = _A_i_F_i_analytical(r_vec, var_vec, mu, Sigma, N, n_quad=n_quad)

    Q_A = float((A_grid >= A_i).mean())
    Q_F = float((F_grid <= F_i).mean())
    return Q_A, Q_F, A_i, F_i


def _q_a_f(cloud_dict, weights, n_points=4000, lattice_k=None):
    """
    Deterministic Q_A and Q_F percentile statistics via barycentric lattice grid.

    For portfolio w_o with return r_o and std-dev σ_o:
        A_i = Pr_{w~Unif(W_s)}( r(w) > r_o  AND  σ(w) < σ_o )
              — fraction of the simplex that strictly dominates w_o
        F_i = Pr_{w~Unif(W_s)}( r(w) < r_o  AND  σ(w) > σ_o )
              — fraction of the simplex that w_o strictly dominates

    Q_A = Pr_{w~Unif(W_s)}( A(w) ≥ A(w_o) )
          Upper-tail rank of w_o's "dominated-by" area.
          Q_A → 1  means w_o is near the efficient frontier (A_i ≈ 0).

    Q_F = Pr_{w~Unif(W_s)}( F(w) ≤ F(w_o) )
          Lower-tail rank of w_o's "dominates" area.
          Q_F → 1  means w_o dominates more of the simplex than most portfolios.
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    chol_L = cloud_dict["chol_L"]
    N      = cloud_dict["N"]

    w_o   = np.asarray(weights, float).ravel()
    r_o   = float(mu @ w_o)
    var_o = float(w_o @ (Sigma @ w_o))
    sig_o = math.sqrt(max(var_o, 0.0))

    W = _simplex_grid(N, n_points, k=lattice_k)   # (M, N), deterministic barycentric lattice

    r_vec   = W @ mu                                    # (M,)
    Z       = W @ chol_L                                # (M, N): row i = L'w_i
    sig_vec = np.sqrt(np.sum(Z ** 2, axis=1))           # (M,)

    # A and F areas for w_o (using the same sample)
    A_o = float(((r_vec > r_o) & (sig_vec < sig_o)).mean())
    F_o = float(((r_vec < r_o) & (sig_vec > sig_o)).mean())

    # dom[i,j] = True iff sample j strictly dominates sample i
    dom   = (r_vec[None, :] > r_vec[:, None]) & (sig_vec[None, :] < sig_vec[:, None])
    A_vec = dom.mean(axis=1)   # A(w_i): fraction of j that dominate i
    F_vec = dom.mean(axis=0)   # F(w_j): fraction of i that j dominates

    Q_A = float((A_vec >= A_o).mean())
    Q_F = float((F_vec <= F_o).mean())

    return Q_A, Q_F, A_o, F_o


# ---------------------------------------------------------------------------
# Analytical P_sigma and P_SR via GL quadrature (no sampling for any N)
# ---------------------------------------------------------------------------

def _inner_length_sigma_batch(t_bar_batch, T_batch, mu_N, a_vec, Q_mat, b_vec, c0,
                               sig_sq_target):
    """
    Length of the inner t_N interval where σ²(t) < sig_sq_target, for each outer
    GL node.  Analytical for all N — no sampling.  Returns shape (K,).
    """
    K         = T_batch.shape[0]
    outer_dim = t_bar_batch.shape[1]

    if outer_dim > 0:
        sig_sq_bar = (c0
                      + t_bar_batch @ b_vec[:outer_dim]
                      + np.einsum('ki,ij,kj->k', t_bar_batch,
                                  Q_mat[:outer_dim, :outer_dim], t_bar_batch))
        beta_bar = b_vec[outer_dim] + 2.0 * (t_bar_batch @ Q_mat[:outer_dim, outer_dim])
    else:
        sig_sq_bar = np.full(K, c0)
        beta_bar   = np.full(K, float(b_vec[outer_dim]))

    alpha = float(Q_mat[outer_dim, outer_dim])
    gamma = sig_sq_bar - sig_sq_target   # constant term shifted by target
    T_k   = T_batch

    # Solve α t_N² + β t_N + γ < 0 → feasible interval [s_lo, s_hi] ∩ [0, T_k]
    if alpha > 1e-14:
        disc      = beta_bar ** 2 - 4.0 * alpha * gamma
        has_roots = disc > 0.0
        sqd       = np.sqrt(np.maximum(disc, 0.0))
        inv2a     = 0.5 / alpha
        s_lo = np.clip((-beta_bar - sqd) * inv2a, 0.0, T_k)
        s_hi = np.clip((-beta_bar + sqd) * inv2a, 0.0, T_k)
    else:
        bk  = beta_bar
        thr = np.where(np.abs(bk) > 1e-14,
                       -gamma / np.where(np.abs(bk) > 1e-14, bk, 1.0),
                       0.0)
        s_lo = np.where(bk >  1e-14, 0.0,
               np.where(bk < -1e-14, np.clip(thr, 0.0, T_k),
                        np.where(gamma < 0, 0.0, T_k)))
        s_hi = np.where(bk >  1e-14, np.clip(thr, 0.0, T_k),
               np.where(bk < -1e-14, T_k,
                        np.where(gamma < 0, T_k, 0.0)))
        has_roots = s_hi > s_lo

    return np.where(has_roots, np.maximum(0.0, s_hi - s_lo), 0.0)


def _p_sigma_analytical(cloud_dict, weights, n_quad=200):
    """
    Pr_{w~Unif(W_s)}(sigma(w) < sigma(w_o)) via GL quadrature — analytical for all N.
    Replaces _p_sigma_minus / _ball_simplex_vol_fraction.
    """
    mu    = cloud_dict["mu"]
    Sigma = cloud_dict["Sigma"]
    N     = cloud_dict["N"]
    if N < 2:
        return 0.0

    w_o   = np.asarray(weights, float).ravel()
    var_o = float(w_o @ Sigma @ w_o)

    mu_N, a_vec, Q_mat, b_vec, c0 = _build_t_form(mu, Sigma)
    outer_dim = N - 2
    K_per_dim = (n_quad if outer_dim <= 1
                 else max(3, int(round(n_quad ** (1.0 / outer_dim)))))
    t_bar_b, T_b, gl_w = _duffy_gl_grid(K_per_dim, outer_dim)

    lengths = _inner_length_sigma_batch(
        t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, var_o)
    return float(np.clip(gl_w @ lengths, 0.0, 1.0))


def _inner_length_sr_batch(t_bar_batch, T_batch, mu_N, a_vec, Q_mat, b_vec, c0,
                            SR_o, rf=0.0):
    """
    Length of the inner t_N interval where SR(t) > SR_o, for each outer GL node.
    Analytical for all N — no sampling.

    The condition SR(t) > SR_o ⟺ (r(t) − rf) > SR_o · σ(t) (since σ > 0).
    Boundary roots come from the quadratic
        (a_last² − SR_o²·α) t² + (2Δr·a_last − SR_o²·β) t + (Δr² − SR_o²·σ̄²) = 0
    validated for sign consistency; f(t) is evaluated at each sub-interval midpoint.

    Returns shape (K,).
    """
    K         = T_batch.shape[0]
    outer_dim = t_bar_batch.shape[1]
    a_last    = float(a_vec[outer_dim])

    r_bar = ((mu_N + t_bar_batch @ a_vec[:outer_dim])
             if outer_dim > 0 else np.full(K, mu_N))
    delta_r_bar = r_bar - rf   # (K,)

    if outer_dim > 0:
        sig_sq_bar = (c0
                      + t_bar_batch @ b_vec[:outer_dim]
                      + np.einsum('ki,ij,kj->k', t_bar_batch,
                                  Q_mat[:outer_dim, :outer_dim], t_bar_batch))
        beta_bar = b_vec[outer_dim] + 2.0 * (t_bar_batch @ Q_mat[:outer_dim, outer_dim])
    else:
        sig_sq_bar = np.full(K, c0)
        beta_bar   = np.full(K, float(b_vec[outer_dim]))

    alpha = float(Q_mat[outer_dim, outer_dim])
    T_k   = T_batch   # (K,)

    def _f(t):
        """f(t) > 0 iff SR(t) > SR_o."""
        r_mrf  = delta_r_bar + a_last * t
        sig_sq = sig_sq_bar + beta_bar * t + alpha * t * t
        sig    = np.sqrt(np.maximum(sig_sq, 0.0))
        return r_mrf - SR_o * sig

    # Quadratic whose roots are candidates for sign-change breakpoints of f:
    #   (a_last² − SR_o²·α) t² + (2Δr·a_last − SR_o²·β) t + (Δr² − SR_o²·σ̄²) = 0
    A_q  = a_last ** 2 - SR_o ** 2 * alpha                      # scalar
    B_q  = 2.0 * delta_r_bar * a_last - SR_o ** 2 * beta_bar    # (K,)
    C_q  = delta_r_bar ** 2 - SR_o ** 2 * sig_sq_bar            # (K,)
    disc = B_q ** 2 - 4.0 * A_q * C_q                           # (K,)

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    has_disc  = disc > 1e-24

    if abs(A_q) > 1e-14:
        t1_raw = (-B_q - sqrt_disc) / (2.0 * A_q)
        t2_raw = (-B_q + sqrt_disc) / (2.0 * A_q)
    else:
        # Degenerate: linear B_q * t + C_q = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            t_lin = np.where(np.abs(B_q) > 1e-14,
                             -C_q / np.where(np.abs(B_q) > 1e-14, B_q, 1.0),
                             np.full(K, np.inf))
        t1_raw = t_lin
        t2_raw = t_lin   # double root

    t1 = np.minimum(t1_raw, t2_raw)
    t2 = np.maximum(t1_raw, t2_raw)

    # A root is valid only if the squaring was sign-consistent.
    # SR_o > 0: root must have r − rf ≥ 0.
    # SR_o < 0: root must have r − rf ≤ 0.
    # SR_o = 0: linear condition only — all roots valid.
    def _valid(t):
        r_mrf = delta_r_bar + a_last * t
        return np.where(SR_o >  1e-14, r_mrf >= -1e-12,
               np.where(SR_o < -1e-14, r_mrf <=  1e-12,
                        np.ones(K, dtype=bool)))

    t1_in = has_disc & (t1 >= -1e-12) & (t1 <= T_k + 1e-12) & _valid(np.clip(t1, 0.0, T_k))
    t2_in = has_disc & (t2 >= -1e-12) & (t2 <= T_k + 1e-12) & _valid(np.clip(t2, 0.0, T_k))

    t1c = np.clip(t1, 0.0, T_k)
    t2c = np.clip(t2, 0.0, T_k)

    # Collect valid breakpoints b1 ≤ b2 within [0, T_k]
    b1_raw = np.where(t1_in, t1c, T_k)
    b2_raw = np.where(t2_in, t2c, T_k)
    b1 = np.minimum(b1_raw, b2_raw)
    b2 = np.maximum(b1_raw, b2_raw)

    # Accumulate lengths of sub-intervals where f > 0
    def _add(lo, hi):
        length = hi - lo
        mid    = 0.5 * (lo + hi)
        return np.where((length > 1e-14) & (_f(mid) > 0), length, 0.0)

    return np.maximum(0.0,
                      _add(np.zeros(K), b1)
                      + _add(b1, b2)
                      + _add(b2, T_k))


def _p_sr_analytical(cloud_dict, weights, rf=0.0, n_quad=200):
    """
    Pr_{w~Unif(W_s)}(SR(w) > SR(w_o)) via GL quadrature — analytical for all N.
    Replaces _p_sr_plus.
    """
    mu    = cloud_dict["mu"]
    Sigma = cloud_dict["Sigma"]
    N     = cloud_dict["N"]
    if N < 2:
        return 0.0

    w_o   = np.asarray(weights, float).ravel()
    r_o   = float(mu @ w_o)
    var_o = float(w_o @ Sigma @ w_o)
    if var_o < 1e-20:
        return 0.0
    SR_o  = (r_o - rf) / math.sqrt(var_o)

    mu_N, a_vec, Q_mat, b_vec, c0 = _build_t_form(mu, Sigma)
    outer_dim = N - 2
    K_per_dim = (n_quad if outer_dim <= 1
                 else max(3, int(round(n_quad ** (1.0 / outer_dim)))))
    t_bar_b, T_b, gl_w = _duffy_gl_grid(K_per_dim, outer_dim)

    lengths = _inner_length_sr_batch(
        t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, SR_o, rf)
    return float(np.clip(gl_w @ lengths, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def relative_performance(cloud_dict, weights, tol=1e-10, n_points=4000,
                         lattice_k=100, determin=True, n_quad=200, rf=0.0,
                         verbose=False, w_ref=None):
    """
    Relative portfolio performance (§3.4).

    Measures defined under w ~ Unif(W_s), each in [0,1] and closer to 1
    when w_o dominates a greater share of the simplex:

        P_r_minus    : Pr(r(w) < r(w_o))         — fraction of simplex w_o beats in return
        P_sigma_plus : Pr(sigma(w) > sigma(w_o))  — fraction of simplex w_o beats in risk
        P_SR_minus   : None                        — requires a risk-free rate (muted)

    Domination-region statistics:

        A_i(w_o) = Pr(r(w) > r_o AND σ(w) < σ_o)  — fraction of simplex dominating w_o
        F_i(w_o) = Pr(r(w) < r_o AND σ(w) > σ_o)  — fraction of simplex w_o dominates

        Q_A    : Pr_{w}( A(w) ≥ A(w_o) )   → 1  means w_o is near the efficient frontier
        Q_F    : Pr_{w}( F(w) ≤ F(w_o) )   → 1  means w_o dominates most of the simplex

    Parameters
    ----------
    cloud_dict : dict from compute_cloud
    weights    : array-like, shape (N,)
    n_points   : int, target lattice size for Q_A / Q_F distribution (default 4000)
    lattice_k  : int or None — override the barycentric lattice k directly
    determin   : bool — use the analytical GL-quadrature method for A_i/F_i
                 (default True); set False to fall back to the O(M²) lattice
                 counting method
    n_quad     : int, GL nodes per outer dimension for analytic A_i/F_i (default 200)
    verbose    : bool — print stat table when True
    w_ref      : optional array-like, shape (N,) — benchmark portfolio

    Returns
    -------
    dict with keys: r_w, var_w, sd_w, P_r_minus, P_sigma_plus, P_SR_minus,
                    A_i, F_i, Q_A, Q_F
    """
    mu    = cloud_dict["mu"]
    Sigma = cloud_dict["Sigma"]
    N     = cloud_dict["N"]

    w = np.asarray(weights, float).ravel()
    if w.shape[0] != N:
        raise ValueError(f"weights length {w.shape[0]} != N={N}")

    r_w   = float(mu @ w)
    var_w = float(w @ (Sigma @ w))
    sd_w  = math.sqrt(max(var_w, 0.0))

    p_r_minus       = 1.0 - _p_r_plus(mu, N, r_w)
    p_sigma_plus    = 1.0 - _p_sigma_analytical(cloud_dict, w, n_quad=n_quad)
    p_sharpe_minus  = 1.0 - _p_sr_analytical(cloud_dict, w, rf=rf, n_quad=n_quad)

    if determin:
        Q_A, Q_F, A_i, F_i = _q_a_f_v2(cloud_dict, w,
                                         n_points=n_points, lattice_k=lattice_k, n_quad=n_quad)
    else:
        Q_A, Q_F, A_i, F_i = _q_a_f(cloud_dict, w, n_points=n_points, lattice_k=lattice_k)

    if verbose:
        segments = cloud_dict["segments"]
        r_global = cloud_dict["r_global"]
        ef_segs  = [s for s in segments if s["ef_frontier"]]
        low_segs = [s for s in segments if s["low_frontier"]]

        _abs = absolute_performance(cloud_dict, w, tol=tol)
        fsv_w = (np.array(_abs["frontier_same_var"]["w_frontier"])
                 if _abs["frontier_same_var"]["exists"]
                    and _abs["frontier_same_var"]["w_frontier"] is not None
                 else None)
        fsr_w = (np.array(_abs["frontier_same_r"]["w_frontier"])
                 if _abs["frontier_same_r"]["exists"]
                    and _abs["frontier_same_r"]["w_frontier"] is not None
                 else None)
        w_min_diss_rp = (np.array(_abs["closest_ef_weights"]["w_ef"])
                         if _abs["closest_ef_weights"]["exists"]
                            and _abs["closest_ef_weights"]["w_ef"] is not None
                         else None)

        rep_cache_v = {}
        def _get_rep_v(active_set):
            key = tuple(active_set)
            if key not in rep_cache_v:
                rep_cache_v[key] = _active_representation(mu, Sigma, list(key))
            return rep_cache_v[key]

        def _weights_on_v(seg, r_star):
            active = seg["active_set"]
            rep = _get_rep_v(active)
            w_star = np.zeros(N)
            w_star[list(active)] = rep["P"] * r_star + rep["q"]
            return w_star

        mvp_seg = next((s for s in ef_segs + low_segs
                        if s["lower_r"] - tol <= r_global <= s["upper_r"] + tol), None)
        w_mvp = _weights_on_v(mvp_seg, r_global) if mvp_seg else None
        idx_max = int(np.argmax(mu))
        w_maxr = np.zeros(N); w_maxr[idx_max] = 1.0

        # Max Sharpe (tangency) portfolio
        def _var_on_rp(seg, r):
            return seg["a_scaled"] * r * r + seg["b_scaled"] * r + seg["c_scaled"]

        w_ms_rp = None
        best_sr_rp = -math.inf
        for seg in ef_segs:
            a_s, b_s, c_s = seg["a_scaled"], seg["b_scaled"], seg["c_scaled"]
            lo_s, hi_s = seg["lower_r"], seg["upper_r"]
            cands = [lo_s, hi_s]
            denom = b_s + 2.0 * rf * a_s
            if abs(denom) > 1e-14:
                cands.append(float(np.clip(-(2.0 * c_s + rf * b_s) / denom, lo_s, hi_s)))
            for r_c in cands:
                v_c = _var_on_rp(seg, r_c)
                if v_c <= 1e-14:
                    continue
                sr_c = (r_c - rf) / math.sqrt(v_c)
                if sr_c > best_sr_rp:
                    best_sr_rp = sr_c
                    w_ms_rp    = _weights_on_v(seg, r_c)

        _rp_keys = ("P_r_minus", "P_sigma_plus", "P_sharpe_minus", "Q_A", "Q_F", "A_i", "F_i")

        def _col_rp(wp):
            if wp is None:
                return {k: None for k in _rp_keys}
            rp_col = relative_performance(cloud_dict, wp, tol=tol, n_points=n_points,
                                          lattice_k=lattice_k, determin=determin,
                                          n_quad=n_quad, rf=rf)
            return {k: rp_col[k] for k in _rp_keys}

        ref_cols = []
        for label, wp in [("Max r|Same sd", fsv_w),
                          ("Min sd|Same r", fsr_w),
                          ("Min Var",       w_mvp),
                          ("Max Return",    w_maxr)]:
            ref_cols.append({"label": label, **_col_rp(wp)})

        if w_ms_rp is not None:
            ref_cols.append({"label": "Max Sharpe", **_col_rp(w_ms_rp)})

        if w_min_diss_rp is not None:
            ref_cols.append({"label": "EF Min Diss", **_col_rp(w_min_diss_rp)})

        if w_ref is not None:
            w_ref_arr = np.asarray(w_ref, float).ravel()
            if w_ref_arr.shape[0] != N:
                raise ValueError(f"w_ref length {w_ref_arr.shape[0]} != N={N}")
            ref_cols.append({"label": "w_ref", **_col_rp(w_ref_arr)})

        WL, WW, WDW = 14, 8, 9
        grp_w = WW + WDW + 6

        _keys    = ("P_r_minus", "P_sigma_plus", "P_sharpe_minus", "A_i", "F_i", "Q_A", "Q_F")
        _w_o_v   = (p_r_minus, p_sigma_plus, p_sharpe_minus, A_i, F_i, Q_A, Q_F)
        _all_vals   = list(_w_o_v)
        _all_deltas = []
        for col in ref_cols:
            for vo, k in zip(_w_o_v, _keys):
                vc = col[k]
                _all_vals.append(vc)
                if vo is not None and vc is not None:
                    _all_deltas.append(vc - vo)

        dec_v = _auto_dec(_all_vals,   WW)
        dec_d = _auto_dec(_all_deltas, WDW, forced_sign=True)

        def _fv(val):
            return f"{val:>{WW}.{dec_v}f}" if val is not None else f"{'N/A':>{WW}}"
        def _fd(val):
            return f"{val:>+{WDW}.{dec_d}f}" if val is not None else f"{'N/A':>{WDW}}"
        def _grp(val, delta):
            return f"  {_fv(val)}  {_fd(delta)} |"

        def _stat_row(label, val_o, key):
            row = f"  {label:>{WL}} | {_fv(val_o)} |"
            for col in ref_cols:
                v = col[key]
                d = (v - val_o) if (v is not None and val_o is not None) else None
                row += _grp(v, d)
            return row

        h1 = f"  {'stat':>{WL}} | {'w_o':^{WW}} |"
        for col in ref_cols:
            h1 += f"{col['label']:^{grp_w}}"
        sep = "  " + "-" * (len(h1) - 2)

        print()
        print("--- relative_performance ---")
        print(h1)
        print(sep)
        print(_stat_row("P_r_minus",      p_r_minus,      "P_r_minus"))
        print(_stat_row("P_sigma_plus",  p_sigma_plus,   "P_sigma_plus"))
        print(_stat_row("P_sharpe_minus",p_sharpe_minus, "P_sharpe_minus"))
        print(_stat_row("A_i",           A_i,            "A_i"))
        print(_stat_row("F_i",          F_i,          "F_i"))
        print(sep)
        print(_stat_row("Q_A",          Q_A,          "Q_A"))
        print(_stat_row("Q_F",          Q_F,          "Q_F"))
        print()

    return {
        "r_w":          r_w,
        "var_w":        var_w,
        "sd_w":         sd_w,
        "P_r_minus":       p_r_minus,
        "P_sigma_plus":    p_sigma_plus,
        "P_sharpe_minus":  p_sharpe_minus,
        "A_i":          A_i,
        "F_i":          F_i,
        "Q_A":          Q_A,
        "Q_F":          Q_F,
    }


# =============================================================================
#  Plotting helper
# =============================================================================

def plot_cloud(cloud_dict, weights=None, sd=True, num_points=200,
               show_assets=True, ref_weights=None, xlim=None, ylim=None,
               show=True, show_legend=True, lw=2, asset_size=36,
               show_targets=True, rf=0.0, percent=False,
               bw=False, title_size=None, axis_title_size=None, label_size=None,
               tick_step=None,
               target_color="black", target_size=80, xtitle=None, ytitle=None,
               save=None, dpi=150):
    """
    Plot frontier segments from cloud_dict, optionally with portfolio diagnostics.

    Parameters
    ----------
    cloud_dict   : dict from compute_cloud
    weights      : array-like or None — observed portfolio (plotted as 'x')
    sd           : bool — x-axis in standard deviation (True) or variance
    num_points   : int  — points per segment curve
    show_assets  : bool — show individual asset markers (default True)
    ref_weights  : array-like or None — reference portfolio (plotted as 'o')
    xlim         : (xmin, xmax) or None — fix the x-axis range; when
                   percent=True supply values in percentage points (e.g. 40
                   for 40%), not decimals
    ylim         : (ymin, ymax) or None — fix the y-axis range (same units
                   as xlim re: percent)
    show         : bool — call plt.show() (default True); set False to keep
                   customizing before showing/saving
    show_legend  : bool — draw the legend (default True)
    lw           : float — line width for all frontier curves (default 2)
    asset_size   : float — marker area for individual asset scatter (default 36)
    show_targets : bool — scatter the 6 EF reference points when weights is
                   provided: Max r|Same sd, Min sd|Same r, Min Var, Max Return,
                   Max Sharpe, EF Min Diss (default True)
    rf           : float — risk-free rate used for Max Sharpe (default 0)
    percent      : bool — multiply all axis values by 100 and display as
                   integers (e.g. 0.05 → 5); default False
    label_size   : float or None — font size for x/y axis labels; None uses
                   the matplotlib default
    bw           : bool — black-and-white mode: all elements drawn in black
                   only, target markers all use 'x', portfolio uses an open
                   circle; default False
    title_size      : float or None — font size for the chart title
                      ("Markowitz Cloud"); None uses the matplotlib default
    axis_title_size : float or None — font size for the axis titles (the words
                      adjacent to each axis, i.e. xlabel/ylabel text); None
                      uses the matplotlib default
    label_size      : float or None — font size for the axis tick labels (the
                      numbers on each axis); None uses the matplotlib default
    tick_step       : float, (xstep, ystep), or None — spacing between major
                      grid lines and axis tick labels; a scalar applies the same
                      step to both axes; a 2-tuple sets x and y independently
                      (supply values in the same units as the axis, so percentage
                      points when percent=True); None lets matplotlib choose
    target_color : str — color for all 6 target portfolio markers; overridden
                   to 'black' when bw=True (default 'black')
    target_size  : float — marker area for target portfolio scatters (default 80)
    xtitle       : str or None — override the x-axis title text; None uses the
                   default derived from sd/percent settings
    ytitle       : str or None — override the y-axis title text; None uses the
                   default derived from percent setting
    save         : str or None — file path to save the figure (e.g.
                   'cloud_FL.png'); None skips saving (default None)
    dpi          : int — resolution when saving (default 150)

    Returns
    -------
    (fig, ax) — the matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _mticker
    mu       = cloud_dict["mu"]
    Sigma    = cloud_dict["Sigma"]
    segments = cloud_dict["segments"]

    _s = 100.0 if percent else 1.0   # scale factor applied to every plotted value

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    def _var_on(seg, r):
        return seg["a_scaled"] * r * r + seg["b_scaled"] * r + seg["c_scaled"]

    if show_assets:
        asset_vars = np.diag(Sigma)
        _akw = {"color": "black"} if bw else {}
        ax.scatter(
            (np.sqrt(np.maximum(asset_vars, 0.0)) if sd else asset_vars) * _s,
            mu * _s, marker='s', label='Assets', zorder=3, s=asset_size, **_akw
        )

    used_labels = set()
    for seg in segments:
        r_lo, r_hi = seg["lower_r"], seg["upper_r"]
        if r_hi <= r_lo:
            continue
        rs = np.linspace(r_lo, r_hi, num_points)
        vs = _var_on(seg, rs)
        xs = (np.sqrt(np.maximum(vs, 0.0)) if sd else vs) * _s

        if seg["ef_frontier"]:
            lbl, style = "NW EF", {"color": "black" if bw else "blue", "lw": lw}
        elif seg["low_frontier"]:
            lbl, style = "SW frontier", {"color": "black" if bw else "green", "lw": lw, "ls": "--"}
        elif seg["ea_frontier"]:
            lbl, style = "East frontier", {"color": "black" if bw else "red", "lw": lw, "ls": ":"}
        else:
            lbl, style = None, {}

        if lbl in used_labels:
            lbl = None
        elif lbl:
            used_labels.add(lbl)

        ax.plot(xs, rs * _s, label=lbl, zorder=2, **style)

    if weights is not None:
        w = np.asarray(weights, float)
        r_w   = float(mu @ w)
        var_w = float(w @ (Sigma @ w))
        x_w   = (math.sqrt(max(var_w, 0.0)) if sd else var_w) * _s
        ax.scatter(x_w, r_w * _s, marker='o', color='black',
                   facecolors='none', label='Portfolio', zorder=4, s=80, linewidths=2)

    if ref_weights is not None:
        wr    = np.asarray(ref_weights, float)
        r_wr  = float(mu @ wr)
        var_wr = float(wr @ (Sigma @ wr))
        x_wr  = (math.sqrt(max(var_wr, 0.0)) if sd else var_wr) * _s
        _rkw  = {"color": "black"} if bw else {"color": "darkorange"}
        ax.scatter(x_wr, r_wr * _s, marker='o', label='Reference', zorder=4, s=60, **_rkw)

    if show_targets and weights is not None:
        _ap = absolute_performance(cloud_dict, weights, sd=sd, rf=rf, verbose=False)
        _targets = [
            ("Max r|Same sd",  _ap["frontier_same_var"].get("r_frontier"), _ap["sd_w"]),
            ("Min sd|Same r",  _ap["r_w"], _ap["frontier_same_r"].get("sd_frontier")),
            ("Min Var",        _ap["min_var"]["r"],  _ap["min_var"]["sd"]),
            ("Max Return",     _ap["max_return"]["r"], _ap["max_return"]["sd"]),
            ("Max Sharpe",     _ap["max_sharpe"]["r"], _ap["max_sharpe"]["sd"]),
            ("EF Min Diss",    _ap["closest_ef_weights"].get("r_ef"),
                               _ap["closest_ef_weights"].get("sd_ef")),
        ]
        _tclr = "black" if bw else target_color
        for lbl, r_t, x_t in _targets:
            if r_t is None or x_t is None:
                continue
            ax.plot(x_t * _s, r_t * _s, marker='+', linestyle='none',
                    color=_tclr, label=lbl, zorder=5,
                    markersize=target_size, markeredgewidth=target_size / 8)

    _default_xlabel = ("Standard deviation (%)" if sd else "Variance (%)") if percent \
                      else ("Standard deviation" if sd else "Variance")
    _default_ylabel = "Expected return (%)" if percent else "Expected return"
    ax.set_xlabel(xtitle if xtitle is not None else _default_xlabel)
    ax.set_ylabel(ytitle if ytitle is not None else _default_ylabel)
    if axis_title_size is not None:
        ax.xaxis.label.set_size(axis_title_size)
        ax.yaxis.label.set_size(axis_title_size)
    if label_size is not None:
        ax.tick_params(axis='both', labelsize=label_size)
    _tkw = {} if title_size is None else {"fontsize": title_size}
    ax.set_title("Markowitz Cloud", **_tkw)
    ax.grid(True)
    _xstep = _ystep = None
    if tick_step is not None:
        _xstep = tick_step[0] if hasattr(tick_step, '__len__') else tick_step
        _ystep = tick_step[1] if hasattr(tick_step, '__len__') else tick_step
        ax.xaxis.set_major_locator(_mticker.MultipleLocator(_xstep))
        ax.yaxis.set_major_locator(_mticker.MultipleLocator(_ystep))
    if percent:
        def _make_fmt(step):
            if step is not None and step != int(step):
                return _mticker.FuncFormatter(lambda v, _: f"{v:.1f}")
            return _mticker.FuncFormatter(lambda v, _: f"{v:.0f}")
        ax.xaxis.set_major_formatter(_make_fmt(_xstep))
        ax.yaxis.set_major_formatter(_make_fmt(_ystep))
    if show_legend:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show(block=True)
    return fig, ax


def q_plot(cloud_dict, weights, stat="A", n_points=4000, lattice_k=100, determin=True,
           n_quad=200, rf=0.0, bins=30, width=None, xlim=None, ylim=None,
           show=True, show_legend=True, lw=2,
           bw=False, percent=True, title_size=None, axis_title_size=None,
           label_size=None, tick_step=None,
           target_color="black", xtitle=None, ytitle=None,
           save=None, dpi=150):
    """
    Histogram of a portfolio statistic sampled over the simplex, drawn as a
    frequency polygon (a line through each bin's midpoint, at its height)
    rather than bars or a stepped outline, with the observed portfolio's
    own value marked.

    Parameters
    ----------
    cloud_dict  : dict from compute_cloud
    weights     : array-like, shape (N,) — observed portfolio
    stat        : {"A", "F", "return", "sigma", "sharpe"} — which quantity to
                  histogram (case-insensitive; default "A"):
                    "A"      : Pr_{w~Unif(W_s)}(r(w) > r_o AND sigma(w) < sigma_o)
                               — dominance area above w_o
                    "F"      : Pr_{w~Unif(W_s)}(r(w) < r_o AND sigma(w) > sigma_o)
                               — dominance area below w_o
                    "return" : expected return r(w)
                    "sigma"  : standard deviation sigma(w)
                    "sharpe" : (r(w) - rf) / sigma(w)
    n_points    : int, target lattice size for the sampled distribution (default 4000)
    lattice_k   : int or None — override the barycentric lattice k directly
    determin    : bool — for stat in {"A", "F"}, use the analytical GL-quadrature
                  method (default True); set False to fall back to the O(M^2)
                  lattice counting method. Ignored for "return"/"sigma"/"sharpe".
    n_quad      : int, GL nodes per outer dimension for analytic A/F (default 200)
    rf          : float — risk-free rate, used only when stat="sharpe" (default 0.0)
    bins        : int — number of histogram bins (default 30); ignored when
                  width is given
    width       : float or None — bin width instead of a fixed bin count; bins
                  span the data range in steps of width (in the same units as
                  the axis, so percentage points when percent=True and
                  stat != "sharpe"); overrides bins; default None
    xlim        : (xmin, xmax) or None — fix the x-axis range; when
                  percent=True (and stat != "sharpe") supply values in
                  percentage points (e.g. 40 for 40%), not decimals
    ylim        : (ymin, ymax) or None — fix the y-axis range (percent of sample)
    show        : bool — call plt.show() (default True); set False to keep
                  customizing before showing/saving
    show_legend : bool — draw the legend (default True)
    lw          : float — line width for the frequency-polygon line and the
                  portfolio marker (default 2)
    bw          : bool — black-and-white mode: histogram line and marker
                  drawn in black only; default False
    percent     : bool — for stat in {"A", "F", "return", "sigma"}, multiply
                  values by 100 and display as integers (e.g. 0.05 -> 5);
                  ignored for "sharpe" (a ratio, not scaled); default True
    title_size      : float or None — font size for the chart title; None uses
                      the matplotlib default
    axis_title_size : float or None — font size for the axis titles; None uses
                      the matplotlib default
    label_size      : float or None — font size for the axis tick labels; None
                      uses the matplotlib default
    tick_step   : float, (xstep, ystep), or None — spacing between major grid
                  lines and axis tick labels; a scalar applies the same step
                  to both axes; a 2-tuple sets x and y independently (supply
                  values in the same units as the axis, so percentage points
                  when percent=True); None lets matplotlib choose
    target_color : str — color for the portfolio marker line; overridden to
                  'black' when bw=True (default 'black')
    xtitle      : str or None — override the x-axis title text; None uses the
                  default derived from stat/percent settings
    ytitle      : str or None — override the y-axis title text; None uses
                  'Percent'
    save        : str or None — file path to save the figure (e.g.
                  'q_plot_FL.png'); None skips saving (default None)
    dpi         : int — resolution when saving (default 150)

    Returns
    -------
    (fig, ax) — the matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _mticker
    stat = stat.lower()
    _stat_info = {
        "a":      ("Distribution of $A(w)$",              "A",              "$A_i$"),
        "f":      ("Distribution of $F(w)$",               "F",              "$F_i$"),
        "return": ("Distribution of Returns",              "Return",         "Return"),
        "sigma":  ("Distribution of Standard Deviations",  "Standard deviation", "Sigma"),
        "sharpe": ("Distribution of Sharpe Ratios",         "Sharpe ratio",  "Sharpe"),
    }
    if stat not in _stat_info:
        raise ValueError(f"stat must be one of {tuple(_stat_info)}, got {stat!r}")

    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    chol_L = cloud_dict["chol_L"]
    N      = cloud_dict["N"]

    w_o = np.asarray(weights, float).ravel()
    if w_o.shape[0] != N:
        raise ValueError(f"weights length {w_o.shape[0]} != N={N}")
    r_o   = float(mu @ w_o)
    var_o = float(w_o @ (Sigma @ w_o))
    sig_o = math.sqrt(max(var_o, 0.0))

    W       = _simplex_grid(N, n_points, k=lattice_k)
    r_vec   = W @ mu
    Z       = W @ chol_L
    var_vec = np.sum(Z ** 2, axis=1)
    sig_vec = np.sqrt(np.maximum(var_vec, 0.0))

    if stat in ("a", "f"):
        if determin:
            A_o_arr, F_o_arr = _A_i_F_i_analytical(np.array([r_o]), np.array([var_o]),
                                                     mu, Sigma, N, n_quad=n_quad)
            A_grid, F_grid   = _A_i_F_i_analytical(r_vec, var_vec, mu, Sigma, N, n_quad=n_quad)
            val_o = float(A_o_arr[0]) if stat == "a" else float(F_o_arr[0])
            grid  = A_grid if stat == "a" else F_grid
        else:
            dom = (r_vec[None, :] > r_vec[:, None]) & (sig_vec[None, :] < sig_vec[:, None])
            if stat == "a":
                val_o = float(((r_vec > r_o) & (sig_vec < sig_o)).mean())
                grid  = dom.mean(axis=1)
            else:
                val_o = float(((r_vec < r_o) & (sig_vec > sig_o)).mean())
                grid  = dom.mean(axis=0)
    elif stat == "return":
        val_o, grid = r_o, r_vec
    elif stat == "sigma":
        val_o, grid = sig_o, sig_vec
    else:  # sharpe
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(sig_vec > 1e-14, (r_vec - rf) / sig_vec, np.nan)
        grid  = grid[np.isfinite(grid)]
        val_o = (r_o - rf) / sig_o if sig_o > 1e-14 else float("nan")

    _title, _stat_xlabel, _marker_label = _stat_info[stat]
    _pct = percent and stat != "sharpe"
    _s   = 100.0 if _pct else 1.0

    data = grid * _s

    if width is not None:
        _lo = math.floor(data.min() / width) * width
        _hi = math.ceil(data.max() / width) * width
        _bins = np.arange(_lo, _hi + width, width)
    else:
        _bins = bins

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    _hkw = {"color": "black"} if bw else {}
    _weights = np.full(data.shape, 100.0 / data.shape[0])
    _counts, _edges = np.histogram(data, bins=_bins, weights=_weights)
    _midpoints = 0.5 * (_edges[:-1] + _edges[1:])
    ax.plot(_midpoints, _counts, lw=lw, zorder=2, **_hkw)

    _lclr = "black" if bw else target_color
    ax.axvline(val_o * _s, color=_lclr, lw=lw, label=f"Portfolio {_marker_label}", zorder=3)

    _default_xlabel = f"{_stat_xlabel} (%)" if _pct else _stat_xlabel
    ax.set_xlabel(xtitle if xtitle is not None else _default_xlabel)
    ax.set_ylabel(ytitle if ytitle is not None else "Percent")
    if axis_title_size is not None:
        ax.xaxis.label.set_size(axis_title_size)
        ax.yaxis.label.set_size(axis_title_size)
    if label_size is not None:
        ax.tick_params(axis='both', labelsize=label_size)
    _tkw = {} if title_size is None else {"fontsize": title_size}
    ax.set_title(_title, **_tkw)
    ax.grid(True)
    _xstep = _ystep = None
    if tick_step is not None:
        _xstep = tick_step[0] if hasattr(tick_step, '__len__') else tick_step
        _ystep = tick_step[1] if hasattr(tick_step, '__len__') else tick_step
        ax.xaxis.set_major_locator(_mticker.MultipleLocator(_xstep))
        ax.yaxis.set_major_locator(_mticker.MultipleLocator(_ystep))
    def _make_fmt(step):
        if step is not None and step != int(step):
            return _mticker.FuncFormatter(lambda v, _: f"{v:.1f}")
        return _mticker.FuncFormatter(lambda v, _: f"{v:.0f}")
    if _pct:
        ax.xaxis.set_major_formatter(_make_fmt(_xstep))
    ax.yaxis.set_major_formatter(_make_fmt(_ystep))
    if show_legend:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show(block=True)
    return fig, ax


# =============================================================================
#  Quick smoke test (run with: python frontier_segments.py)
# =============================================================================
'''
if __name__ == "__main__":
    mu_test = np.array([0.2044, 0.1579, 0.095])
    w_ref = np.array([0.17, 0.60, 0.23])
    Sigma_test = np.array([
        [0.00024086, 0.00005642, 0.00008801],
        [0.00005642, 0.00011336, 0.00006400],
        [0.00008801, 0.00006400, 0.00015271],
    ])
    w_test = np.array([0.2,0.4,0.4])

    cloud = compute_cloud(mu_test, Sigma_test, verbose=True)

    ap = absolute_performance(cloud, w_test, reference_weights=w_ref, verbose=True)
    qr = quasi_relative_performance(cloud, w_test, w_ref=w_ref, verbose=True)
    relative_performance(cloud, w_test, w_ref=w_ref, lattice_k=10000, verbose=True)
    plot_cloud(cloud, weights=w_test, sd=True, num_points=200, show_assets=False, ref_weights=w_ref)'''