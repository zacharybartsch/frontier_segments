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


_GMV_ENUM_MAX = 16   # exhaustive active-set search up to this many assets


def _long_only_gmv(mu, Sigma, tol=1e-12):
    """
    Global minimum-variance portfolio under the long-only constraint:

        min  w' Sigma w    s.t.  1'w = 1,  w >= 0

    This is the (sigma_m^2, r_m) that partitions the west frontier into the
    NW efficient frontier (r >= r_m) and the SW frontier (r < r_m).

    This is NOT the unconstrained GMV, w = Sigma^-1 1 / (1' Sigma^-1 1). The
    two coincide only when the unconstrained solution is already non-negative.
    Otherwise the unconstrained return B/C lies off the long-only frontier
    entirely and must not be used as the anchor: doing so mislabels part of
    the efficient frontier as SW, reports a "minimum variance" allocation that
    is not actually minimum variance, and can hand the critical-line walk a
    starting active set with no feasible r interval at all.

    For N <= _GMV_ENUM_MAX the active set is found by exact enumeration. For
    larger N a primal active-set iteration is used: drop the most negative
    weight until feasible, then admit any asset whose marginal variance
    (Sigma w)_j sits below the budget multiplier lambda (a KKT violation).

    Returns (var, r, active) with `active` a sorted list of asset indices.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    N = len(mu)

    def _solve(idx):
        """Equality-constrained GMV on `idx`; (weights, var) or (None, None)."""
        C = Sigma[np.ix_(idx, idx)]
        e = np.ones(len(idx))
        try:
            x = np.linalg.solve(C, e)
        except np.linalg.LinAlgError:
            return None, None
        d = float(e @ x)
        if not np.isfinite(d) or abs(d) < 1e-18:
            return None, None
        return x / d, 1.0 / d

    if N <= _GMV_ENUM_MAX:
        best = (math.inf, None, None)
        for k in range(1, N + 1):
            for S in itertools.combinations(range(N), k):
                idx = np.array(S, dtype=int)
                w_a, var = _solve(idx)
                if w_a is None or np.min(w_a) < -tol:
                    continue
                if var < best[0]:
                    w = np.zeros(N)
                    w[idx] = w_a
                    best = (float(var), float(mu @ w), sorted(int(i) for i in S))
        if best[2] is not None:
            return best

    active = set(range(N))
    for _ in range(50 * N + 100):
        idx = np.array(sorted(active), dtype=int)
        w_a, var = _solve(idx)
        if w_a is None:
            if len(active) <= 1:
                break
            active.discard(int(idx[-1]))
            continue
        if np.min(w_a) < -tol:
            active.discard(int(idx[int(np.argmin(w_a))]))
            continue
        w = np.zeros(N)
        w[idx] = w_a
        g = Sigma @ w
        lam = float(np.mean(g[idx]))
        cand, gain = None, tol
        for j in range(N):
            if j not in active and lam - float(g[j]) > gain:
                gain, cand = lam - float(g[j]), j
        if cand is None:
            return float(var), float(mu @ w), sorted(int(i) for i in idx)
        active.add(cand)

    idx = np.array(sorted(active), dtype=int)
    w_a, var = _solve(idx)
    if w_a is None:                      # fully degenerate: least-variance asset
        j = int(np.argmin(np.diag(Sigma)))
        return float(Sigma[j, j]), float(mu[j]), [j]
    w = np.zeros(N)
    w[idx] = w_a
    return float(var), float(mu @ w), sorted(int(i) for i in idx)


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

    # Anchor the walk at the LONG-ONLY global minimum-variance point. This is
    # the (sigma_m^2, r_m) that partitions the west frontier into the NW
    # efficient frontier (r >= r_global) and the SW frontier (r < r_global).
    #
    # Using the unconstrained GMV return B/C here instead is wrong whenever the
    # unconstrained solution has a negative weight: it anchors the split at a
    # return that is not the long-only minimum, mislabels the band between the
    # two as SW when it is genuinely efficient, reports a "minimum variance"
    # reference allocation that is not minimum variance, and (via the active
    # set below) can produce weights that violate w >= 0 or leave no feasible
    # r interval at all.
    #
    # The GMV active set is feasible by construction, so it is also the correct
    # starting active set for the critical-line walk in both directions.
    _gmv_var, r_global, start_active = _long_only_gmv(mu, Sigma)

    if not calc_ef and not calc_low:
        return [], r_global

    if len(start_active) == 1:
        # _active_representation needs >= 2 assets (a singleton has D == 0).
        # Pair the GMV asset with whichever partner keeps r_global feasible at
        # the least variance.
        _i = start_active[0]
        _best = None
        for _j in range(N):
            if _j == _i:
                continue
            try:
                _rep_j = _active_representation(mu, Sigma, sorted([_i, _j]))
            except ValueError:
                continue
            if _rep_j["r_lo"] - tol <= r_global <= _rep_j["r_hi"] + tol:
                _v = _rep_j["a"] * r_global ** 2 + _rep_j["b"] * r_global + _rep_j["c"]
                if _best is None or _v < _best[0]:
                    _best = (_v, sorted([_i, _j]))
        if _best is not None:
            start_active = _best[1]

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
                         verbose=False, reference_weights=None, reference=True):
    """
    Absolute portfolio performance relative to the frontier (§3.2).

    Parameters
    ----------
    cloud_dict        : dict returned by compute_cloud
    weights           : array-like, shape (N,)
    sd                : bool — report in standard-deviation units (True) or variance
    reference_weights : optional array-like, shape (N,) — benchmark portfolio;
                        when provided AND reference=True, verbose output includes
                        r and sd of this portfolio alongside deltas relative to weights
    reference         : bool — when verbose=True, include the 6 standard reference-
                        portfolio columns (and reference_weights, if given) in the
                        printed table (default True, matching prior behavior). Set
                        False to print only weights' own r/sd/sharpe/weights row.
                        Does not affect the returned dict, which always includes
                        frontier_same_var/frontier_same_r/nearest_ef/
                        closest_ef_weights/min_var/max_return/max_sharpe — those
                        are cheap to compute regardless and plot_cloud relies on
                        them being present.

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
        if reference:
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
                                rf=0.0, verbose=False, w_ref=None, reference=True):
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
    w_ref      : optional array-like, shape (N,) — benchmark portfolio; scored
                 (and included in verbose output) only when reference=True
    reference  : bool — compute/score w_ref and, when verbose=True, print the
                 6 standard reference-portfolio columns (default True, matching
                 prior behavior). Set False to compute/print only weights' own
                 rho/gamma measures — ref_rho_r/ref_rho_sigma/ref_gamma_r/
                 ref_gamma_sigma/ref_gamma_sharpe/dissim_w_ref are then None
                 regardless of w_ref.

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

    # ---- w_ref (optional; only scored when reference=True) ------------------
    _ref_qrp     = None
    w_ref_arr    = None
    dissim_w_ref = None
    if reference and w_ref is not None:
        w_ref_arr = np.asarray(w_ref, float).ravel()
        if w_ref_arr.shape[0] != N:
            raise ValueError(f"w_ref length {w_ref_arr.shape[0]} != N={N}")
        _ref_qrp     = quasi_relative_performance(cloud_dict, w_ref_arr, sd=sd, tol=tol, rf=rf)
        dissim_w_ref = 0.5 * float(np.sum(np.abs(w - w_ref_arr)))

    # ---- verbose output -----------------------------------------------------
    if verbose:
        ref_cols = []
        if reference:
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
        if ref_cols:
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


def _check_containment(A_i, F_i, p_r_minus, p_sigma_plus, label="w_o", tol=1e-9):
    """
    Set containment the estimates must respect:

        F = {r < r_o} and {sigma > sigma_o}  =>  F_i <= P_sigma_plus
                                             and F_i <= P_r_minus
        A = {r > r_o} and {sigma < sigma_o}  =>  A_i <= 1 - P_r_minus
                                             and A_i <= 1 - P_sigma_plus

    These hold for the underlying quantities but NOT automatically for the
    estimates, because the four are produced by three different methods at
    different effective resolutions. A violation does not mean any single
    estimator is wrong -- it measures how far apart they are. Warn, never
    raise: the pipeline is expected to trip this until the levels are
    converged (see METHODS_ROADMAP.md).

    Returns the worst slack (negative = violated).
    """
    # Reference-column labels carry a pipe ("Max r|Same sd"), and the research
    # driver parses verbose output by splitting on "|" -- an unsanitized label
    # here lands in the exported table as a spurious row.
    label = str(label).replace("|", "/")
    checks = {}
    if F_i is not None:
        if p_sigma_plus is not None:
            checks["F_i <= P_sigma_plus"] = p_sigma_plus - F_i
        if p_r_minus is not None:
            checks["F_i <= P_r_minus"] = p_r_minus - F_i
    if A_i is not None:
        if p_r_minus is not None:
            checks["A_i <= 1 - P_r_minus"] = (1.0 - p_r_minus) - A_i
        if p_sigma_plus is not None:
            checks["A_i <= 1 - P_sigma_plus"] = (1.0 - p_sigma_plus) - A_i
    if not checks:
        return None
    worst_key = min(checks, key=checks.get)
    worst = checks[worst_key]
    if worst < -tol:
        print(f"relative_performance: containment violated for {label} -- "
              f"{worst_key} fails by {-worst:.6f}. The estimates disagree by "
              f"at least this much; see METHODS_ROADMAP.md.")
    return float(worst)
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


_GL_NODE_BUDGET = 10_000_000   # default total outer GL nodes per evaluation
_GL_K_CAP       = 64           # per-dimension cap; past this nothing moves
_GL_CHUNK       = 500_000      # outer nodes materialized at once


def _gl_nodes_per_dim(budget, outer_dim, k_cap=_GL_K_CAP):
    """
    Largest K with K**outer_dim <= budget, capped at k_cap.

    A budget must be a ceiling, never a floor. The previous rule,
    max(3, round(budget ** (1/d))), could round UP past the budget and then
    refuse to go below 3 per dimension, so at large d it demanded far more
    nodes than asked for -- 3**18 is 387 million, a 56 GB grid, whatever
    budget was requested. Here K is only ever reduced to fit.

    Convergence is in K, not in the total, so the cap keeps low-dimensional
    cases cheap: at outer_dim=3 the measures are already converged by K~30,
    and spending the whole budget there would buy nothing.
    """
    if outer_dim <= 0:
        return 1
    if outer_dim == 1:
        return int(max(2, min(budget, 200_000)))
    K = max(2, int(budget ** (1.0 / outer_dim)))
    while (K + 1) ** outer_dim <= budget:
        K += 1
    while K > 2 and K ** outer_dim > budget:
        K -= 1
    return int(min(K, k_cap))


def _duffy_gl_chunks(K_per_dim, outer_dim, chunk=_GL_CHUNK):
    """
    Same nodes as _duffy_gl_grid, yielded in blocks instead of all at once.

    The full grid is (K**d, d) float64 plus several (K**d,) companions, so it
    is the grid -- not the integrand -- that sets the memory ceiling and
    crashes at high d. Decomposing the flat node index with unravel_index
    reproduces each block exactly, so memory is O(chunk) regardless of K**d
    and the node budget can be chosen for accuracy rather than for RAM.
    """
    import math
    if outer_dim == 0:
        yield np.empty((1, 0)), np.ones(1), np.ones(1)
        return

    nodes, gl_w = np.polynomial.legendre.leggauss(K_per_dim)
    u1d = (nodes + 1.0) / 2.0
    w1d = gl_w / 2.0

    if outer_dim == 1:
        yield u1d.reshape(-1, 1), 1.0 - u1d, w1d * math.factorial(2)
        return

    K_total = K_per_dim ** outer_dim
    shape   = (K_per_dim,) * outer_dim
    for start in range(0, K_total, chunk):
        stop  = min(start + chunk, K_total)
        multi = np.unravel_index(np.arange(start, stop, dtype=np.int64), shape)
        u_flat = np.stack([u1d[m] for m in multi], axis=1)
        w_flat = np.prod(np.stack([w1d[m] for m in multi], axis=1), axis=1)

        n     = stop - start
        t_bar = np.zeros((n, outer_dim))
        cum   = np.ones(n)
        for i in range(outer_dim):
            t_bar[:, i] = u_flat[:, i] * cum
            cum = cum * (1.0 - u_flat[:, i])

        jac = np.ones(n)
        for j in range(outer_dim - 1):
            jac *= (1.0 - u_flat[:, j]) ** (outer_dim - 1 - j)

        yield t_bar, cum, w_flat * jac * math.factorial(outer_dim + 1)


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
# ---------------------------------------------------------------------------
# Exact 2-D dominance counting on the deterministic lattice
# ---------------------------------------------------------------------------

def _count_smaller_before(v):
    """
    For each i, the number of j < i with v[j] < v[i].  O(M log M) by bottom-up
    vectorized merge counting (the standard offline inversion-count sweep).

    The array is padded to a power of two with a sentinel strictly above every
    real value, so padding is never counted as "smaller" and its own results
    are discarded.  Per merge level the rows are kept sorted, and a per-row
    offset makes the flattened left halves globally sorted so a single
    np.searchsorted serves every row at once.
    """
    M = v.shape[0]
    if M < 2:
        return np.zeros(M, dtype=np.int64)

    P_    = 1 << (M - 1).bit_length()
    SENT  = int(v.max()) + 1
    BIG   = SENT + 1
    val   = np.full(P_, SENT, dtype=np.int64); val[:M] = v
    idx   = np.arange(P_, dtype=np.int64)
    res_p = np.zeros(P_, dtype=np.int64)

    b = 1
    while b < P_:
        val = val.reshape(-1, 2 * b); idx = idx.reshape(-1, 2 * b)
        nrow = val.shape[0]
        left, right = val[:, :b], val[:, b:]
        off     = (np.arange(nrow, dtype=np.int64) * BIG)[:, None]
        lf      = (left + off).ravel()
        rf_     = (right + off).ravel()
        rowbase = np.repeat(np.arange(nrow, dtype=np.int64) * b, b)

        # side="left" counts strictly-smaller (the statistic); side="right"
        # gives the stable merge position (equal values keep left first).
        res_p[idx[:, b:].ravel()] += np.searchsorted(lf, rf_, side="left") - rowbase
        dest = (np.searchsorted(lf, rf_, side="right") - rowbase)                + np.tile(np.arange(b, dtype=np.int64), nrow)
        rows = np.repeat(np.arange(nrow, dtype=np.int64), b)

        nv = np.empty_like(val); ni = np.empty_like(idx)
        nv[rows, dest] = right.ravel()
        ni[rows, dest] = idx[:, b:].ravel()
        mask = np.ones((nrow, 2 * b), dtype=bool); mask[rows, dest] = False
        nv[mask] = left.ravel()
        ni[mask] = idx[:, :b].ravel()
        val, idx = nv, ni
        b *= 2

    return res_p[:M]


def _dominance_counts(r, s):
    """
    For each i, the exact count of j with r[j] > r[i] AND s[j] < s[i].

    Sorting r descending with s descending as the secondary key makes ties
    self-handling: within a group of equal r the points are ordered by
    non-increasing s, so no earlier member of the group has strictly smaller
    s and equal-r pairs contribute nothing.  Dense ranking of s keeps the
    s-comparison strict as well.
    """
    M = r.shape[0]
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    order    = np.lexsort((-s, -r))               # last key is primary
    s_sorted = s[order]
    ranks    = np.searchsorted(np.unique(s_sorted), s_sorted).astype(np.int64)
    out = np.empty(M, dtype=np.int64)
    out[order] = _count_smaller_before(ranks)
    return out


_SOBOL_DEFAULT = 1 << 22      # 4,194,304 points
_SOBOL_CHUNK   = 1 << 19      # points materialized at once


def _sobol_pow2(n_points):
    """Largest power of two at or below n_points, floored at 2**10."""
    import math
    m = max(10, int(math.floor(math.log2(max(float(n_points), 1.0)))))
    return m, 1 << m


def _sobol_block(N, start, n):
    """
    Points [start, start+n) of the Sobol sequence, mapped to the simplex.

    Sorted spacings of d = N-1 coordinates are exactly uniform on the
    (N-1)-simplex: for sorted u, the gaps (u_1, u_2-u_1, ..., 1-u_last) are
    Dirichlet(1,...,1). The sequence is unscrambled, so there is no seed and
    no randomness -- the same N and index give the same point forever.
    """
    from scipy.stats import qmc
    eng = qmc.Sobol(d=N - 1, scramble=False)
    if start:
        eng.fast_forward(start)
    u = np.sort(eng.random(n), axis=1)
    W = np.empty((n, N))
    W[:, 0] = u[:, 0]
    if N > 2:
        W[:, 1:N - 1] = np.diff(u, axis=1)
    W[:, N - 1] = 1.0 - u[:, -1]
    return W


def _sobol_moments(N, M, mu, chol_L, chunk=_SOBOL_CHUNK, verbose=False):
    """
    (r, sigma) for the first M Sobol points, weights generated in blocks and
    discarded so peak memory is the two vectors plus one block.

    Why Sobol rather than a uniform lattice: A(w) depends on w only through
    (r, sigma), so the integrand is two-dimensional however many assets there
    are. Sobol's low-dimensional projections are well distributed by the
    (t,m,s)-net property, while a uniform lattice's are not -- an N-asset grid
    projected onto the (r, sigma) plane clumps badly. Measured against
    independently known values (exact convex-hull P_r_minus, converged
    quadrature P_sigma_plus), Sobol at 262,144 points beat a 735,471-point
    lattice by two to three orders of magnitude at N=9.

    Note the first Sobol point is the origin, which maps to a simplex vertex;
    that is one point out of M and is left in place.
    """
    r = np.empty(M)
    s = np.empty(M)
    for start in range(0, M, chunk):
        n = min(chunk, M - start)
        W = _sobol_block(N, start, n)
        r[start:start + n] = W @ mu
        s[start:start + n] = np.sqrt(np.maximum(np.sum((W @ chol_L) ** 2, axis=1), 0.0))
        del W
    if verbose:
        print(f"_sobol_moments: {M:,} Sobol points, N={N} "
              f"(deterministic, unscrambled; no seed)")
    return r, s


def _a_f_lattice(r_vec, sig_vec):
    """
    A and F for EVERY point of the set, exactly, in O(M log M).

        A(w_i) = #{j : r_j > r_i AND sigma_j < sigma_i} / M
        F(w_i) = #{j : r_j < r_i AND sigma_j > sigma_i} / M

    Point-set agnostic -- it counts whatever (r, sigma) pairs it is given.

    F is A under (r, sigma) -> (-r, -sigma), so one kernel serves both.
    These are the same definitions the quadrature path targets; here they are
    evaluated on the lattice itself rather than integrated, so the only error
    is the lattice's own resolution -- no quadrature node budget is involved
    and a thin dominating region cannot silently collapse to zero.
    """
    M = r_vec.shape[0]
    if M == 0:
        return np.zeros(0), np.zeros(0)
    A = _dominance_counts(r_vec, sig_vec).astype(float) / M
    F = _dominance_counts(-r_vec, -sig_vec).astype(float) / M
    return A, F


def _a_f_point(r_vec, sig_vec, r_p, sig_p):
    """
    A and F for one arbitrary portfolio measured against the same point set.
    O(M). Used for w_o and the reference portfolios, which are exact critical
    line objects and are not members of the point set.
    """
    M = r_vec.shape[0]
    A = float(((r_vec > r_p) & (sig_vec < sig_p)).sum()) / M
    F = float(((r_vec < r_p) & (sig_vec > sig_p)).sum()) / M
    return A, F
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


def _p_sigma_analytical(cloud_dict, weights, n_quad=_GL_NODE_BUDGET):
    """
    Pr_{w~Unif(W_s)}(sigma(w) < sigma(w_o)) via GL quadrature — analytical for all N.
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
    K_per_dim = _gl_nodes_per_dim(n_quad, outer_dim)

    total = 0.0
    for t_bar_b, T_b, gl_w in _duffy_gl_chunks(K_per_dim, outer_dim):
        lengths = _inner_length_sigma_batch(
            t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, var_o)
        total += float(gl_w @ lengths)
    return float(np.clip(total, 0.0, 1.0))


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


def _p_sr_analytical(cloud_dict, weights, rf=0.0, n_quad=_GL_NODE_BUDGET):
    """
    Pr_{w~Unif(W_s)}(SR(w) > SR(w_o)) via GL quadrature — analytical for all N.
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
    K_per_dim = _gl_nodes_per_dim(n_quad, outer_dim)

    total = 0.0
    for t_bar_b, T_b, gl_w in _duffy_gl_chunks(K_per_dim, outer_dim):
        lengths = _inner_length_sr_batch(
            t_bar_b, T_b, mu_N, a_vec, Q_mat, b_vec, c0, SR_o, rf)
        total += float(gl_w @ lengths)
    return float(np.clip(total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _reference_portfolios(cloud_dict, w, tol=1e-10, rf=0.0, w_ref=None):
    """
    The reference allocations that head the columns of the performance tables,
    in display order, each tagged with whether it sits on the NW efficient
    frontier by construction.

    Every one of these is an exact frontier object produced by the critical
    line algorithm -- a lattice cannot supply them, because a grid of
    barycentric points does not contain the frontier, only points near it.
    Shared by both A/F methods so the two paths head identical columns.

    Returns a list of {"label", "w", "is_ef"}; "w" may be None when a
    reference allocation does not exist for this cloud.
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    N      = cloud_dict["N"]
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

    w_ref_arr = None
    if w_ref is not None:
        w_ref_arr = np.asarray(w_ref, float).ravel()
        if w_ref_arr.shape[0] != N:
            raise ValueError(f"w_ref length {w_ref_arr.shape[0]} != N={N}")


    refs = [{"label": "Max r|Same sd", "w": fsv_w,  "is_ef": False},
            {"label": "Min sd|Same r", "w": fsr_w,  "is_ef": False},
            {"label": "Min Var",       "w": w_mvp,  "is_ef": True},
            {"label": "Max Return",    "w": w_maxr, "is_ef": True}]
    if w_ms_rp is not None:
        refs.append({"label": "Max Sharpe",  "w": w_ms_rp,       "is_ef": True})
    if w_min_diss_rp is not None:
        refs.append({"label": "EF Min Diss", "w": w_min_diss_rp, "is_ef": True})
    if w_ref_arr is not None:
        refs.append({"label": "w_ref",       "w": w_ref_arr,     "is_ef": False})
    return refs


def relative_performance(cloud_dict, weights, tol=1e-10, n_points=_SOBOL_DEFAULT,
                         rf=0.0, verbose=False, w_ref=None, reference=False,
                         method="analytic", n_quad_p=_GL_NODE_BUDGET,
                         lattice_k=None, determine=None, n_quad=None, coarse=None):
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

    A_i, F_i, Q_A and Q_F are computed by one of two methods. Neither changes
    any definition above.

      method="count" (default) — A and F are evaluated on the deterministic
        barycentric lattice itself by exact 2-D dominance counting, one
        O(M log M) sweep for all M points. Error is the lattice's own
        resolution and nothing else; it shrinks predictably in k and cannot
        collapse a thin dominating region to a spurious zero.
        A(w) is then a function OF the lattice and Q_A a rank WITHIN it, so
        numerator and population are one consistent object.

      method="quad" — the legacy iterated Gauss-Legendre path. The inner
        1-D length is exact, but the outer (N-2)-dimensional integrand is
        non-smooth (kinks plus a compact support boundary), so Gauss-Legendre
        has no advantage there, and n_quad is a TOTAL node budget spread as
        n_quad**(1/(N-2)) per dimension — only 3 nodes per dimension once
        N >= 7. At that resolution thin dominating regions integrate to
        exactly zero. Retained for validating levels at high n_quad and for
        reproducing earlier results; not recommended for Q_A or Q_F.

      determine=False is the historic spelling of method="count".

    The reference-portfolio columns are exact critical-line-algorithm
    objects under both methods (see _reference_portfolios); a lattice can
    measure a distribution but cannot locate a frontier.

    Implementation notes (QA_boundary_refinement_spec.md — none of this
    changes any of the above definitions; it only changes how they are
    computed):
      - A_i/F_i for w_o are always computed from a single fused quadrature
        pass (both from the same node sweep) with dominance-membership
        skips applied where a single threshold's Q_A/Q_F is being computed
        directly (determine=True, coarse-refined or not).
      - reference=True additionally computes the 6 reference-portfolio
        columns (mirroring absolute_performance): 4 of them (global min
        variance, global max return, max Sharpe, min-D EF allocation) sit
        exactly on the NW EF by construction, so A_i=0 and Q_A=1 for them
        trivially (nothing has both strictly higher return AND strictly
        lower risk than a non-dominated point) — no lattice work needed for
        those two values. F is NOT free for these (F=0 does not
        characterize the whole EA). Only 2 reference points (the
        max-return-conditional and min-variance-conditional allocations)
        plus w_o itself and an optional w_ref can genuinely sit off the EF
        and need real A/Q_A computation. F always needs real computation
        for every column (up to 8), so F always uses a shared F-area
        distribution computed once over the lattice and reused (bisection-
        style counting) rather than resampled per column — this replaces
        the previous implementation's recursive relative_performance() call
        per reference portfolio, which resampled the full lattice every
        time. Since the count of real A-thresholds is small (<=4) even
        under reference=True, `coarse` (see below) applies to each of them
        independently rather than being ignored under reference=True.
      - coarse (default False): when set, Q_A/Q_F for a threshold are
        computed via boundary refinement instead of full-lattice
        evaluation — see _q_percentile_boundary. False/None = direct
        evaluation (exact, with dominance skips); True = default coarse
        resolution; int = explicit coarse resolution. Applies to whichever
        real A-thresholds exist (always, regardless of reference); does NOT
        apply to F when reference=True (F's shared-distribution regime
        already amortizes across all its thresholds, so per-threshold
        boundary refinement would not help there — see the reuse-regime
        discussion above). Boundary refinement is an opt-in speed/accuracy
        trade — validate a chosen coarse resolution with
        validate_coarse_halving before trusting its results for reporting.

    Parameters
    ----------
    cloud_dict : dict from compute_cloud
    weights    : array-like, shape (N,)
    n_points   : int, target lattice size for Q_A / Q_F distribution (default
                 1_000_000); ignored when lattice_k is given
    lattice_k  : int or None — override the barycentric lattice k directly;
                 None (default) auto-derives k from n_points
    determine  : bool — use the analytical GL-quadrature method for A_i/F_i
                 (default True); set False to fall back to the O(M²) lattice
                 counting method (reference/coarse are ignored in that path)
    n_quad     : int, GL nodes per outer dimension for analytic A_i/F_i (default 200)
    verbose    : bool — print the stat table when True; independent of
                 `reference` (verbose controls printing, reference controls
                 what gets computed — see spec)
    w_ref      : optional array-like, shape (N,) — benchmark portfolio;
                 only scored when reference=True
    reference  : bool — additionally compute the 6 reference-portfolio
                 columns (i-vi: max-return-conditional, min-variance-
                 conditional, global min variance, global max growth, max
                 Sharpe, min-D EF allocation), mirroring
                 absolute_performance(); default False (cheap path — only
                 w_o's own measures)
    coarse     : bool or int — boundary-refinement coarse resolution for
                 per-threshold Q_A/Q_F (see notes above); default False
                 (exact direct evaluation)

    Returns
    -------
    dict with keys: r_w, var_w, sd_w, P_r_minus, P_sigma_plus, P_SR_minus,
                    A_i, F_i, Q_A, Q_F, and (only when reference=True)
                    reference_columns — list of per-reference-portfolio dicts
                    with the same stat keys plus "label".
    """
    mu     = cloud_dict["mu"]
    Sigma  = cloud_dict["Sigma"]
    chol_L = cloud_dict["chol_L"]
    N      = cloud_dict["N"]

    w = np.asarray(weights, float).ravel()
    if w.shape[0] != N:
        raise ValueError(f"weights length {w.shape[0]} != N={N}")

    r_w   = float(mu @ w)
    var_w = float(w @ (Sigma @ w))
    sd_w  = math.sqrt(max(var_w, 0.0))

    # ── deprecated arguments ────────────────────────────────────────────
    for _name, _val in (("lattice_k", lattice_k), ("determine", determine),
                        ("n_quad", n_quad), ("coarse", coarse)):
        if _val is not None and _val is not False:
            print(f"relative_performance: {_name!r} is ignored -- the lattice and "
                  f"quadrature paths for A/F were replaced by Sobol. Use n_points.")

    _method = str(method).lower()
    if _method not in ("analytic", "sobol"):
        raise ValueError(f"method must be 'analytic' or 'sobol', got {method!r}")

    # ── the point set ───────────────────────────────────────────────────
    # One Sobol set serves everything. Its first M//4 points are a strict
    # prefix, so the coarse run used for the convergence estimate costs an
    # extra sweep but no extra point generation.
    _m_pow, M = _sobol_pow2(n_points)
    r_vec, sig_vec = _sobol_moments(N, M, mu, chol_L, verbose=verbose)
    M4 = M // 4
    r4, s4 = r_vec[:M4], sig_vec[:M4]

    def _sharpe(rv, sv):
        return np.divide(rv - rf, sv, out=np.full_like(rv, -np.inf), where=sv > 0)

    # ── P measures ──────────────────────────────────────────────────────
    # Default keeps the exact and quadrature forms: P_r_minus is an exact
    # convex-hull polytope volume, and P_sigma_plus / P_sharpe_minus are
    # Gauss-Legendre over an analytic reduction. Both are supported by the
    # existing literature and converge at high node counts. method="sobol"
    # recomputes all three from the point set instead, as a cross-check.
    if _method == "sobol":
        sr_w = (r_w - rf) / sd_w if sd_w > 0 else -np.inf
        p_r_minus      = float((r_vec < r_w).mean())
        p_sigma_plus   = float((sig_vec > sd_w).mean())
        p_sharpe_minus = float((_sharpe(r_vec, sig_vec) < sr_w).mean())
        p_r_minus4     = float((r4 < r_w).mean())
        p_sigma_plus4  = float((s4 > sd_w).mean())
        p_sharpe_minus4 = float((_sharpe(r4, s4) < sr_w).mean())
    else:
        p_r_minus       = 1.0 - _p_r_plus(mu, N, r_w)
        p_sigma_plus    = 1.0 - _p_sigma_analytical(cloud_dict, w, n_quad=n_quad_p)
        p_sharpe_minus  = 1.0 - _p_sr_analytical(cloud_dict, w, rf=rf, n_quad=n_quad_p)
        p_r_minus4 = p_sigma_plus4 = p_sharpe_minus4 = None   # not M-dependent

    # ── A, F, Q_A, Q_F -- always Sobol ──────────────────────────────────
    A_all, F_all = _a_f_lattice(r_vec, sig_vec)
    A_i, F_i = _a_f_point(r_vec, sig_vec, r_w, sd_w)
    Q_A = float((A_all >= A_i).mean())
    Q_F = float((F_all <= F_i).mean())

    A4, F4 = _a_f_lattice(r4, s4)
    A_i4, F_i4 = _a_f_point(r4, s4, r_w, sd_w)
    Q_A4 = float((A4 >= A_i4).mean())
    Q_F4 = float((F4 <= F_i4).mean())
    del A4, F4

    # ── convergence: |stat(M) - stat(M/4)| ──────────────────────────────
    # Not a bound. It is how far the statistic moved over a fourfold increase
    # in points, which is the honest empirical handle on an estimator whose
    # error is deterministic rather than random.
    convergence = {
        "M": M, "M_coarse": M4,
        "A_i": abs(A_i - A_i4), "F_i": abs(F_i - F_i4),
        "Q_A": abs(Q_A - Q_A4), "Q_F": abs(Q_F - Q_F4),
        "P_r_minus":      None if p_r_minus4 is None else abs(p_r_minus - p_r_minus4),
        "P_sigma_plus":   None if p_sigma_plus4 is None else abs(p_sigma_plus - p_sigma_plus4),
        "P_sharpe_minus": None if p_sharpe_minus4 is None else abs(p_sharpe_minus - p_sharpe_minus4),
    }

    # ── gauge: Sobol's P_r_minus against its exactly known value ────────
    _p_r_sobol = float((r_vec < r_w).mean())
    sobol_gauge = {"P_r_minus_exact": 1.0 - _p_r_plus(mu, N, r_w),
                   "P_r_minus_sobol": _p_r_sobol,
                   "gap": _p_r_sobol - (1.0 - _p_r_plus(mu, N, r_w)),
                   "M": M}
    _check_containment(A_i, F_i, p_r_minus, p_sigma_plus, "w_o")

    # ── reference columns ───────────────────────────────────────────────
    ref_cols = None
    if reference:
        _none = {k: None for k in ("P_r_minus", "P_sigma_plus", "P_sharpe_minus",
                                   "Q_A", "Q_F", "A_i", "F_i")}
        ref_cols = []
        for _r in _reference_portfolios(cloud_dict, w, tol=tol, rf=rf, w_ref=w_ref):
            wp = _r["w"]
            if wp is None:
                ref_cols.append({"label": _r["label"], **_none})
                continue
            r_p  = float(mu @ wp)
            sd_p = math.sqrt(max(float(wp @ (Sigma @ wp)), 0.0))
            A_p, F_p = _a_f_point(r_vec, sig_vec, r_p, sd_p)
            if _method == "sobol":
                sr_p = (r_p - rf) / sd_p if sd_p > 0 else -np.inf
                prm = float((r_vec < r_p).mean())
                psp = float((sig_vec > sd_p).mean())
                psm = float((_sharpe(r_vec, sig_vec) < sr_p).mean())
            else:
                prm = 1.0 - _p_r_plus(mu, N, r_p)
                psp = 1.0 - _p_sigma_analytical(cloud_dict, wp, n_quad=n_quad_p)
                psm = 1.0 - _p_sr_analytical(cloud_dict, wp, rf=rf, n_quad=n_quad_p)
            ref_cols.append({
                "label": _r["label"],
                "P_r_minus": prm, "P_sigma_plus": psp, "P_sharpe_minus": psm,
                "A_i": A_p, "F_i": F_p,
                "Q_A": float((A_all >= A_p).mean()),
                "Q_F": float((F_all <= F_p).mean()),
            })
            _check_containment(A_p, F_p, prm, psp, _r["label"])
    del A_all, F_all, r_vec, sig_vec


    if verbose:
        ref_cols_print = ref_cols if ref_cols is not None else []
        WL, WW, WDW = 14, 8, 9
        grp_w = WW + WDW + 6

        _keys    = ("P_r_minus", "P_sigma_plus", "P_sharpe_minus", "A_i", "F_i", "Q_A", "Q_F")
        _w_o_v   = (p_r_minus, p_sigma_plus, p_sharpe_minus, A_i, F_i, Q_A, Q_F)
        _all_vals   = list(_w_o_v)
        _all_deltas = []
        for col in ref_cols_print:
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
            for col in ref_cols_print:
                v = col[key]
                d = (v - val_o) if (v is not None and val_o is not None) else None
                row += _grp(v, d)
            return row

        h1 = f"  {'stat':>{WL}} | {'w_o':^{WW}} |"
        for col in ref_cols_print:
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
        _g, _c = sobol_gauge, convergence
        print(f"  M = {_g['M']:,} Sobol points (deterministic, unscrambled)")
        print(f"  gauge: P_r_minus exact {_g['P_r_minus_exact']:.6f} vs Sobol "
              f"{_g['P_r_minus_sobol']:.6f}, gap {_g['gap']:+.2e}")
        _cs = "  ".join(f"{k} {_c[k]:.1e}" for k in ("A_i", "F_i", "Q_A", "Q_F"))
        print(f"  convergence |stat(M) - stat(M/4)|:  {_cs}")
        print()

    result = {
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
        "M":             M,
        "convergence":   convergence,
        "sobol_gauge":   sobol_gauge,
    }
    if reference:
        result["reference_columns"] = ref_cols
    return result


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
    target_size  : float — marker AREA in points^2 for the target-portfolio
                   markers, on the same scale as the w_o and reference
                   markers (default 80; they use 60)
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
        # target_size is an AREA (points^2), matching scatter's s= used by the
        # w_o and reference markers above. Line2D.markersize is a LINEAR size
        # in points, so it takes the square root -- passing the area straight
        # through drew crosses about ten times too large.
        _tms = math.sqrt(max(float(target_size), 0.0))
        for lbl, r_t, x_t in _targets:
            if r_t is None or x_t is None:
                continue
            ax.plot(x_t * _s, r_t * _s, marker='+', linestyle='none',
                    color=_tclr, label=lbl, zorder=5,
                    markersize=_tms, markeredgewidth=max(_tms / 8.0, 0.5))

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


def q_plot(cloud_dict, weights, stat="A", n_points=_SOBOL_DEFAULT,
           lattice_k=None, determine=None, n_quad=None, method=None,
           rf=0.0, bins=30, width=None, xlim=None, ylim=None,
           show=True, show_legend=True, lw=2,
           bw=False, percent=True, title_size=None, axis_title_size=None,
           label_size=None, tick_step=None,
           target_color="black", xtitle=None, ytitle=None,
           save=None, dpi=150, graph=True, stats=False, stats_save=None, stats_sheet=None):
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
    n_points    : int, target lattice size for the sampled distribution
                  (default 1_000_000); ignored when lattice_k is given
    lattice_k   : int or None — override the barycentric lattice k directly;
                  None (default) auto-derives k from n_points
    determine   : bool — for stat in {"A", "F"}, use the analytical GL-quadrature
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
    graph       : bool — draw and (optionally) show/save the histogram
                  (default True); set False to skip plotting entirely, e.g.
                  when only stats=True is wanted
    stats       : bool — export an Excel sheet describing the plotted
                  distribution (default False): N, Min, the 10th-90th
                  percentiles (deciles), Max, Mean, Std Dev, Skewness, and
                  Kurtosis (excess, i.e. normal=0), in that order, computed
                  on the same values used for the histogram (so in percent
                  units when percent=True, matching the plot). Independent
                  of graph — either can be True/False regardless of the
                  other.
    stats_save  : str or None — file path for the stats Excel workbook; None
                  (default) derives it from save by replacing its extension
                  with '.xlsx' (e.g. 'q_plot_FL.png' -> 'q_plot_FL.xlsx'),
                  or falls back to 'q_plot_stats.xlsx' if save is also None.
                  If the target file already exists, the stats are added as
                  a NEW SHEET in that workbook (mode='a') rather than
                  overwriting it — pass the same stats_save across multiple
                  q_plot calls (e.g. one per state in a loop) to collect
                  them all into one workbook, one sheet per call, matching
                  the pattern used for performance-measure exports
                  elsewhere in this project. Only used when stats=True.
    stats_sheet : str or None — sheet name for the stats table; None
                  (default) uses "{STAT} distribution" (e.g. "A distribution").
                  Give each call in a loop a distinct name (e.g. an
                  f"{abb} {stat}" including the state) so they don't
                  overwrite each other's sheet when sharing one stats_save
                  workbook. Truncated to Excel's 31-character sheet-name
                  limit. Only used when stats=True.

    Returns
    -------
    (fig, ax) — the matplotlib Figure and Axes, or (None, None) if graph=False.
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

    for _nm, _v in (("lattice_k", lattice_k), ("determine", determine),
                    ("n_quad", n_quad), ("method", method)):
        if _v is not None:
            print(f"q_plot: {_nm!r} is ignored -- the point set is Sobol now. "
                  f"Use n_points.")

    _m_pow, M = _sobol_pow2(n_points)
    r_vec, sig_vec = _sobol_moments(N, M, mu, chol_L)
    var_vec = sig_vec * sig_vec

    if stat in ("a", "f"):
        # Same dominance counting as relative_performance, over the same kind
        # of point set, so the histogram and the reported A_i/F_i agree.
        A_grid, F_grid = _a_f_lattice(r_vec, sig_vec)
        A_o, F_o = _a_f_point(r_vec, sig_vec, r_o, sig_o)
        val_o = A_o if stat == "a" else F_o
        grid  = A_grid if stat == "a" else F_grid
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

    if stats:
        import pandas as pd
        from scipy.stats import skew, kurtosis
        from pathlib import Path

        _stat_rows = [("N", int(data.shape[0])), ("Min", float(data.min()))]
        for _p in range(10, 91, 10):
            _stat_rows.append((f"P{_p}", float(np.percentile(data, _p))))
        _stat_rows.append(("Max", float(data.max())))
        _stat_rows.append(("Mean", float(data.mean())))
        _stat_rows.append(("Std Dev", float(data.std())))
        _stat_rows.append(("Skewness", float(skew(data))))
        _stat_rows.append(("Kurtosis", float(kurtosis(data))))  # excess kurtosis, normal=0

        _value_col = f"{_stat_xlabel} (%)" if _pct else _stat_xlabel
        _stats_df = pd.DataFrame(_stat_rows, columns=["Statistic", _value_col])

        if stats_save is not None:
            _stats_path = stats_save
        elif save is not None:
            _stats_path = str(Path(save).with_suffix(".xlsx"))
        else:
            _stats_path = "q_plot_stats.xlsx"
        _sheet_name = str(stats_sheet if stats_sheet is not None
                           else f"{stat.upper()} distribution")[:31]

        if Path(_stats_path).exists():
            # Append as a new sheet in the existing workbook (matching the
            # single-workbook / one-sheet-per-call pattern used elsewhere in
            # this project for performance-measure exports) instead of
            # overwriting the whole file -- if_sheet_exists="replace" only
            # replaces a sheet with the SAME name (e.g. a rerun), leaving
            # every other state/stat's sheet intact.
            with pd.ExcelWriter(_stats_path, engine="openpyxl", mode="a",
                                 if_sheet_exists="replace") as _writer:
                _stats_df.to_excel(_writer, index=False, sheet_name=_sheet_name)
        else:
            _stats_df.to_excel(_stats_path, index=False, sheet_name=_sheet_name)

    if not graph:
        return None, None

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