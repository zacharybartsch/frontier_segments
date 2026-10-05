# Methods roadmap — simplex measure evaluation

**Status: planning document. Nothing below item 0 is implemented.**
**Written 2026-10-04 against commit `52c3997`, working tree clean.**

---

## 0. Read this before touching anything

If you are picking this up fresh, the package has changed substantially and
several widely-repeated descriptions of it are now wrong. Verify against the
code, not against memory or older docs.

### What is already done and committed

| commit | change |
|---|---|
| `2866c6b` | Frontier anchored at the **long-only** GMV, not the unconstrained B/C |
| `1fbbb5f` | `A_i`, `F_i`, `Q_A`, `Q_F` computed by **exact lattice dominance counting**, not quadrature |
| `ba3af61` | `n_quad_p` (default 10,000,000) for the P-measures; node budget is a **ceiling**; Duffy grid **chunked** |
| `52c3997` | Accuracy note corrected — at N=9 the quadrature is converged, the lattice is the laggard |

Current defaults in `relative_performance`:

```
method="count"   n_quad_p=10_000_000   n_quad=200   n_points=1_000_000
lattice_k=None   determine=True        coarse=False
```

### Current method, per measure — do not assume one method covers all

| measure | region | method | status |
|---|---|---|---|
| `P_r_minus` | half-space ∩ simplex (a **polytope**) | exact volume, convex hull | **exact** |
| `P_sigma_plus`, `P_sharpe_minus` | simplex minus ellipsoid interior (**one smooth boundary**) | Duffy + Gauss-Legendre, `n_quad_p` | converged; ~0.002pp at N≤8, ~0.1pp at N=9 |
| `A_i`, `F_i`, `Q_A`, `Q_F` | half-space ∩ ellipsoid interior ∩ simplex (**two intersecting boundaries, corners, slivers**) | exact dominance counting on the barycentric lattice, O(M log M) | exact *on the lattice*; lattice itself is coarse |

The method is chosen by the region's geometry, not by taste. One method
everywhere is what the package used to do, and it put `A_i` off by 6× and
`P_sigma_plus` off by up to 33pp.

### Traps

- **Two different node counts.** `k` = lattice subdivision (weights are
  multiples of 1/k), giving `points` = C(k+N−1, N−1). `K` = Gauss-Legendre
  nodes *per dimension* over N−2 dimensions, set by `n_quad_p`. Unrelated.
  Conversations about this code routinely conflate them.
- **`n_quad` is not `n_quad_p`.** `n_quad` (200) governs only the legacy
  `method="quad"` path, where it multiplies by ~10⁶ lattice points.
  `n_quad_p` (10M) governs the P-measures, evaluated once per portfolio.
  Raising `n_quad` to 10M would attempt ~10¹² evaluations.
- **`_dominance_masks` and `coarse` are fossils** of the quadrature era. They
  saved real work when each `A(w)` was a separate integration. Under counting
  `A(w)` is produced for every point by one sorted pass, so they save nothing.
  Item 11 below subsumes them; delete rather than rewire.
- **`_A_i_F_i_analytical` is retained deliberately** — as the characterization
  result and as a validation instrument at high `n_quad`. It is *not* the
  production estimator. Do not restore it as one.
- **site-packages is a manual copy, not an editable install.** The research
  scripts import from site-packages. It has silently served a stale module
  more than once. `pip install -e .` would end this.

### Known-open defect

The three estimators are not mutually consistent. `F ⊆ {σ > σ_o}` requires
`F_i ≤ P_sigma_plus`; this currently **fails**:

| state | `F_i` (lattice) | `P_sigma_plus` (quadrature) | violation |
|---|---|---|---|
| MO | 0.387963 | 0.380790 | −0.007 |
| NJ | 0.371068 | 0.229037 | **−0.142** |

Not a bug in either estimator. `F_i` at the default lattice is badly
unconverged — New Jersey's reported 0.371 extrapolates to ~0.242 — while the
quadrature is converged. Mixing a converged estimator with an unconverged one
breaks relations that hold for the underlying quantities. Items 2, 3 and 6
below address this.

---

## Settled — no conflict, not yet implemented

**1. Stream the weight matrix.** Generate lattice blocks, compute `r` and `σ`,
discard. `W` is M×N and is the single largest object; nothing downstream needs
it. Roughly 4× more points for the same memory.

**2. Add containment assertions.** `F_i ≤ P_sigma_plus` and
`A_i ≤ 1 − P_r_minus`. Cheap, permanent, catches the defect above before it
reaches a table.

**3. Report the `P_r_minus` calibration gauge.** Compute `P_r_minus` on the
lattice as well as exactly; print the gap. A per-state empirical measure of
that lattice's discretization error, for one extra comparison. NJ at k=16
reads 0.976 against the exact 0.9928 — 1.7pp — which immediately tells you
`A_i` and `F_i` carry errors of that order.

**4. Set the default point budget.** OPEN — needs a decision. 1/k < 0.0001 is
unreachable above N=3. At a 10⁸-point ceiling: 1/k ≈ 0.003 (N=5), 0.013 (N=7),
0.021 (N=9).

---

## Mutually exclusive — decisions still required

### 5. How `A(w)` is computed across the lattice

- **5a. Exact sweep** (what is committed). Exact integer counts, O(M) memory,
  caps M near 10⁸.
- **5b. Binned streaming histogram.** Bin `(r, σ)` into a G×G grid, prefix-sum
  it, classify from lookups. Memory O(G²) **independent of M**. Counts become
  approximate — binning error adds to lattice error.

### 6. How `A_i` and `F_i` levels are reported

- **6a. Raw count** at the chosen k. Biased, but with known sign and a bound.
- **6b. Richardson-extrapolated** across 2–3 k. Error is clean O(1/k), so
  fitting `a + b/k` recovers the limit. Costs ~1.2× if you extrapolate
  *downward* from the k you were already running. **Fails silently when
  convergence is non-monotone** — Illinois's lattice `P_sigma_plus` read
  0.628, 0.657, 0.634 at k = 14, 20, 26. Mitigate by fitting ≥3 k and checking
  the limit agrees across pairs; disagreement is the diagnostic.
- **6c. Raw count plus a printed error bar** (boundary-straddling fraction, or
  the item-3 gauge). Honest and conservative; no correction.

6b and 6c combine. 6a is the do-nothing.

### 7. Whether `A_i` uses the quadrant identity

Three constraints (two marginals, and the four quadrants summing to 1) against
four unknowns leaves one degree of freedom, so computing one joint gives the
other:

> **A_i = 1 − P_r_minus − P_sigma_plus + F_i**

- **7a. Count both** `A_i` and `F_i`. Internally consistent; both carry
  lattice error.
- **7b. Count `F_i`, derive `A_i`.** Halves the level work and `A_i` inherits
  `P_r_minus`'s exactness — but mixes estimators, which is the disease behind
  the item-0 defect. On current numbers the identity gives NJ 0.1492 against a
  reported 0.0154, and MO 0.0238 against 0.0011.

Either way the identity is worth computing **as a check**: on a single common
population it holds by construction, so the residual measures how far apart
the estimators are.

---

## Evaluation reduction — MEASURED AND REJECTED (2026-10-04)

**Do not rebuild these without reading this section.** Items 9, 10 and 11 were
estimated at 4-25x combined. Built and measured, they deliver nothing against
a vectorized sweep. The estimates were wrong in a way worth recording.

**Item 11, built and validated, is slower than what it replaces.** The
four-row partition below is correct -- verified exact against brute force
across 20 configurations, N = 3-7, k = 4-14, 8 thresholds each. But:

| state | N | k | M | lines | tree | sweep |
|---|---|---|---|---|---|---|
| GA | 5 | 40 | 135,751 | 12,341 | 0.32s | **0.14s** |
| LA | 7 | 18 | 134,596 | 33,649 | 0.94s | **0.17s** |
| NJ | 9 | 12 | 125,970 | 50,388 | 1.28s | **0.19s** |

Per-line cost in Python is ~25 us; the NumPy sweep costs ~1 us per POINT. To
win, a line must cost under (points per line) x 1 us -- 11 us at GA, 2.5 us
at NJ. Vectorizing the tree removes the interpreter overhead but the ceiling
is still points-per-line, which is 2.5 at N=9. The operations item 11
eliminates were the cheapest ones available.

**Item 10 certifies 0.000%-0.223% of points** (median 0.03%), against actual
A_i + F_i of 10%-64%:

| IL | LA | AL | KY | MS | TX | GA | MO | NJ | OK | AR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002% | 0.000% | 0.007% | 0.223% | 0.021% | 0.042% | 0.000% | 0.006% | 0.169% | 0.042% | 0.031% |

A face certifies only if EVERY portfolio on it qualifies. Into A that needs
max_{i in S} Sigma_ii < var_o -- every individual asset less volatile than the
diversified observed portfolio. Georgia: var_o = 0.00127 against asset
variances [0.00117, 0.0236, 0.0025, 0.0381, 0.0161], so only one asset
qualifies and A-certification can fire only on singleton faces. Into F it
needs max_{i in S} mu_i < r_o -- any one good asset kills the face. The mass
is at support 5-7, where faces are mixed. Uniformity over a face is the wrong
thing to ask for.

**Item 9 inherits both failures.** Its subtree bounds are looser than the
exact face min/max that just failed, and its ceiling is the same
points-per-line factor that item 11 could not cash in. Not built.

**What the exercise did establish**, and the reason the partition below is
kept: it is the correct way to reason about a line, it is validated, and it
documents exactly which region resists certification.

### The remaining lever is M, not evaluation reduction

Q_A's accuracy is bounded by k; k is bounded by the O(M) memory of the
dominance sweep. The way to improve it is item 5b (binned streaming, memory
O(G^2) independent of M), not by touching fewer points. Nothing in this
section changes that.

---

## The line partition (kept as theory, not as an optimization)

**11. Line-wise closed-form classification — do this first.**
Along any line in the simplex, `r` is linear and `σ²` is quadratic, so the
image in (r, σ²) is a parabola — already the segment representation
(`a_scaled`, `b_scaled`, `c_scaled`).

**Canonical statement.** Parametrise the line by r and solve `σ(r) = σ_o`,
giving roots r₁ < r₂ (σ < σ_o strictly between them, since the parabola opens
upward in (r, σ²)). Three numbers — r_o, r₁, r₂ — partition the entire line:

| region | r vs r_o | σ vs σ_o | verdict |
|---|---|---|---|
| r < min(r_o, r₁) | below | above | **greater A** — certify |
| (r₂, r_o), when r₂ < r_o | below | above | **greater A** — certify |
| (max(r_o, r₁), r₂) | above | below | **lesser A** — certify |
| r > r_o with σ > σ_o | above | above | **mixed** — not certifiable |

Classify the whole line and count its points by arithmetic. Note the second
row: on a line whose σ rises again past r₂ while r is still below r_o, that
upper stretch certifies too — easy to miss if you reason from slope signs
rather than from the roots.

Do **not** reason about this via the sign of dσ/dr. The roots already encode
which side of the vertex you are on, and slope arguments repeatedly produced
intervals on the wrong side of r_o. The only region that resists
certification is r > r_o together with σ > σ_o — better return, worse risk,
no dominance relation in either direction.

The innermost enumeration loop *is* a line (it moves mass between the last two
assets), so it never needs to execute point-by-point. Exact and
unconditional, not bound-dependent.

**The factor is k/(N−1), not k.** Points per line = (k+N−1)/(N−1):

| | N=5, k=67 | N=7, k=26 | N=9, k=16 |
|---|---|---|---|
| points per line | 17.8 | 5.3 | **3.0** |

At N=9 a line is three points long, so line-wise methods save little there.
This is the same shape of disappointment that killed RLE and the staggered
lattices: the mechanism scales with k, and k is smallest exactly where N is
largest.

For `Q_A` there is no closed form — `Ã(r, σ)` is not a parabola — but it is
monotone in each argument, so the crossing `Ã = A_o` along a line admits
bisection: O(log k) evaluations instead of k, with everything jumped over
certified by monotonicity.

**Warm start the bisection from the sub-face.** For S′ ⊂ S the superset's
min-variance frontier lies weakly below the subset's (φ²_S(r) ≤ φ²_S′(r)), and
`Ã` increases in σ, so the subset's crossing of `Ã = A_o` brackets the
superset's. Walking up the subset lattice carries the bracket along, cutting
bisection steps perhaps 2–3× per line. This is bracketing only — it does not
certify points, because the superset face also opens up high-σ / low-r
territory the subset never reached.

**10. Face-based certification — strongest at high N.**
`σ²` is a convex quadratic, so its minimum over face(S) is the S-restricted
long-only GMV (`_long_only_gmv` already computes this). Precompute all 2^N
faces — 512 QPs at N=9, trivial. The return side needs no QP: max/min of `r`
over face(S) is max/min μ over S.

**Certify against the A-level curve, not against σ_o.** A face's image in
(r, σ) space lies entirely above its own min-variance frontier, and `Ã`
increases in σ. So:

> If a face's min-variance frontier lies entirely above the level curve
> {Ã = A_o}, every point of that face has A > A_o — certify the whole face.

This is strictly stronger than testing min σ over the face against σ_o,
because it targets A directly instead of routing through the F-quadrant.

**Test the edge skeleton, not the face.** Slice a face at constant r: the
cross-section is a polytope whose vertices lie on the face's *edges*. `σ²` is
convex, so its maximum over that slice is attained at a vertex — hence on an
edge. Therefore:

> If every edge of face(S) spanning return level r has σ < σ_o, then every
> point of the slice does, and the whole slice dominates w_o.

So a face is certified by checking only its 1-skeleton — C(|S|,2) parabolas
instead of a QP. This is both **cheaper** (parabola roots versus a quadratic
program) and **stronger** (it certifies the face's interior, not merely its
frontier). The same convexity argument runs in the other direction for the
"greater A" side.

Combined with item 11, this means face certification and line classification
use the same primitive: the roots of `σ(r) = σ_o` on an edge.

**Search order.** Certification propagates downward by pure set containment:
face(S′) ⊂ face(S) for S′ ⊂ S, so certifying a large face already covers
every sub-face — there is no separate extension step, the sub-faces' points
are *in* face(S). Exploit this in the search, not in the classification: test
large faces first and skip all their sub-faces on success; on failure skip
their supersets, since a sub-face that fails means the superset cannot
certify wholesale. That prunes the 2^N precompute — immaterial at N=9 (512
QPs), significant at N=20 (10⁶).

The upward direction classifies nothing. "A < A_o on a sub-face" extends to
the superset only along the frontier, which is measure zero; the superset
contains new territory where A > A_o. Its value is the warm start in item 11.

Why it pays: at small k and large N almost no lattice points have full
support. N=9, k=16, 735,471 points by support size —

| support | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| points | 57,330 | 171,990 | 252,252 | 180,180 | 57,915 | **6,435** |

Only 0.87% use all nine assets. Counts are closed form: C(k−1, s−1) points per
support set, C(N,s) sets of that size. **This is the only mechanism here whose
efficiency improves as N grows.** How much certifies is state-dependent —
Georgia's σ_o sits at the 2.6th percentile so most faces certify; New Jersey's
at the 60th so few do.

**9. Branch-and-bound over the composition tree — opportunistic.**
At a node with budget B over m remaining assets, `r` is linear so the subtree
range is O(1) (all of B on the highest-μ remaining asset, then the lowest).
Certified subtree size is **C(B+m−1, m−1)**, closed form — add the count,
never descend. Extends to `A` because `Ã` is monotone: a subtree's (r, σ) box
gives bounds on `Ã`, and an interval clear of A_o certifies the subtree.
`Q_A` needs only class *counts*, never per-point values, so this works for the
rank statistic too.

Frictions: `σ²` bounds are quadratic and loose (cheap eigenvalue bounds prune
poorly; tight bounds cost a small QP per node), and bounding `Ã` needs the
global (r, σ) distribution to exist first.

**Caveat on combining 9 and 10.** They are different decompositions — 10
partitions by *support*, 9 by *prefix*. A tree node has fixed nonzero weights
plus free budget, so its σ range involves cross terms the face table does not
cover. Pick one as primary; use the other opportunistically.

### Spitball, stacked

Touches saved as a multiple of M. `Q_A` is the binding case; `A_i`/`F_i` do
better because they are closed form per line with no bisection.

| | N=5 | N=7 | N=9 |
|---|---|---|---|
| 11 alone — `A_i`/`F_i`, closed form | 17.8× | 5.3× | 3.0× |
| 11 alone — `Q_A`, warm-started bisection | ~6× | ~2× | ~1.5× |
| + 10, face certification | ~2× more | ~2× more | ~2× more |
| + 9, outer B&B | ~2× more | ~1.5× more | ~1.3× more |
| **combined, `Q_A`** | **~25×** | **~6×** | **~4×** |

Translated into achievable spacing at ~10⁹ touches (the saving compounds —
larger k means longer lines):

| N | current 1/k | with the stack |
|---|---|---|
| 5 | 0.0149 | **~0.001** |
| 7 | 0.0385 | ~0.006 |
| 9 | 0.0625 | ~0.010 |

Roughly 6–15× better spacing. N=5 reaches 0.001; N=7 and N=9 do not.

**Three caveats on this table.**

1. Item 10's 2× is a placeholder, and it is the least estimable number here.
   Actual yield depends on where each state's σ_o sits against the face
   frontiers — near-total for Georgia (σ_o at the 2.6th percentile),
   near-nothing for New Jersey (60th). **Measure this before building it.**
2. Item 9's σ bounds are quadratic and loose; 1.3–2× may be optimistic.
3. The factors are multiplicative only if the mechanisms prune *different*
   points, and they overlap — a certified face contains whole lines. Treat
   the combined row as an upper-ish estimate, not arithmetic.

---

## Rejected or superseded — do not revisit without new argument

- **Staggered / offset lattices.** Rejected. For `A_o` alone (a fixed region)
  averaging offset grids is sound and standard. For `Q_A` it is not: `A` is
  defined *relative to the lattice*, so shifting changes both the population
  and the function being ranked. The R runs estimate R different quantities
  sharing a common k-driven bias, which averaging does not remove.
- **Union of a lattice with its midpoints.** This *is* the lattice at 2k. The
  point count rises by ~2^(N−1), which is 104× at N=9, k=16→32 — not the
  "less than double" that holds only at N=2. Pooling counts across separate
  runs is also invalid: `A` is population-relative, so the runs classify
  against different references.
- **Run-length encoding the greater/lesser lists.** Moot. If you are only
  counting you need two integers, not lists. Compression would have been
  ≈ (k+N−1)/(N−1) anyway — ~3× at N=9.
- **8a (wire `_dominance_masks` to the counting path) and 8b (per-point
  gradient-bound B&B).** Dropped. Certification covers A_i + F_i of points,
  7%–64% by state, but under counting `A(w)` is already an O(1) lookup, so
  replacing it with two comparisons saves nothing. Item 11 achieves the same
  monotonicity argument at line level, where it pays combinatorially.

---

## Facts worth not re-deriving

- `Ã` depends on w **only through (r, σ)**. The problem is effectively
  2-dimensional regardless of N. Most of the leverage above follows from this.
- `A(w) = 0` **iff** w is on the efficient frontier. Under counting, all four
  EF-resident reference portfolios return `A_i = 0` on their own, with no
  special-casing — a free correctness check.
- `A` is **convex** (half-space ∩ ellipsoid interior ∩ simplex). `F` is **not**
  (ellipsoid *exterior*). Convexity of `A` does not make counting exact — cells
  straddling the boundary are counted wholly in or out regardless — but it does
  make the O(1/k) error model reliable, which is what 6b depends on.
- Lattice points = C(k+N−1, N−1), **not** k^N. The sum-to-one constraint
  removes a dimension. At 1pp spacing and N=3 that is 5,151 points, not 10⁶.
- The lattice includes the simplex's **vertices, edges and faces** — the pure
  and two-asset portfolios, where the frontier's extreme allocations live. A
  random draw almost surely never lands on them.
- Reference allocations are exact critical-line-algorithm objects under every
  option here. A lattice measures a distribution; it cannot locate a frontier.
  Nothing above changes that.

---

## Decisions outstanding

1. Item 4 — default point budget.
2. Item 5 — 5a exact sweep, or 5b binned streaming.
3. Item 6 — 6a / 6b / 6c (6b and 6c combine).
4. Item 7 — 7a count both, or 7b derive `A_i` from the identity.
5. ~~Whether to implement 11, 10, 9~~ — **settled 2026-10-04: measured and
   rejected, see the evaluation-reduction section.** The live successor
   question is whether to take item 5b (binned streaming) to raise M.
