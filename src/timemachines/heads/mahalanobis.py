"""Streaming anomaly detection on the prediction parade.

(Ported from the skaters proto branch `proto/anomaly-mahalanobis`; skaters
is now a dependency — the parade state contract `state["z"]` is the API
boundary between the packages.)

The parade reduces an arbitrary stream to — under calibration — a stationary
vector ``z_t ~ N(0, Sigma)``: the standard-normal surprises of the arriving
point under the predictions issued 1..k steps ago. That is exactly the
homogeneous input classical multivariate-Gaussian outlier detection assumes.
The forecaster is the normaliser; detection reduces to the textbook robust
Mahalanobis case, applied to *trailing predictions* instead of raw features.

``mahalanobis`` wraps a parade-wrapped skater (e.g. :func:`skaters.laplace`)
and maintains a streaming robust estimate of the location ``mu`` and scatter
``Sigma`` of z. Each tick it scores

    d2 = (z - mu)' Sigma^{-1} (z - mu),

and converts d2 to a p-value through an *empirical null* (in the spirit of
Efron 2004): under the theoretical null d2 ~ chi^2_k, but in practice the
z-components are often strongly correlated across horizons (on smooth streams
the forecasts issued at t-1..t-k are nearly identical, so Sigma is
near-singular) and the shrinkage floor then collapses the effective degrees
of freedom below k. Rather than pretend, the wrapper tracks the running mean
``m`` and variance ``v`` of d2 and matches the null to a scaled chi-square
``c * chi^2_nu`` by two-moment (Welch--Satterthwaite) matching, ``c = v/2m``,
``nu = 2m^2/v``, seeded at the theoretical moments ``(m, v) = (k, 2k)``. The
p-value in ``state["pvalue"]`` is therefore approximately Uniform(0,1) on
well-specified data by construction, whatever the effective rank.

Scoring happens *before* any update — a point must not defend itself — and
every update (mu, Sigma, and the null moments alike) is Huberised: ticks whose
d2 exceeds the ``guard_p`` empirical-null quantile are downweighted by
``q_guard / d2``, the online cousin of reweighted MCD. Without this, anomalies
contaminate the estimates and inflate them (masking), hiding later anomalies.
Pure self-exclusion has its own failure mode — after a structural break every
tick looks anomalous forever — so a run of more than ``adapt_after``
consecutive guarded ticks is treated as a changepoint and updates resume at
full weight. In practice an adaptive base forecaster absorbs level shifts
itself within a few ticks, so ``state["run"]`` scales with the *forecaster's
adaptation time*: isolated spikes read as point outliers (run 1-2, the echo
tick coming from the outlier polluting the next forecast), structural breaks
as longer runs.

Two structural facts make the scores meaningful from the very first matured
tick: the parade standardises each margin, so the identity matrix is not an
arbitrary initialisation of Sigma but the *calibrated prior* (and the
``shrink`` toward it a principled Ledoit--Wolf-style target); and the null
moments are seeded at their exact theoretical values.

Entries of ``state["d2"]``/``state["pvalue"]`` are ``None`` until every
horizon of the parade has matured (the first k observations) and on any tick
where a horizon's z is unavailable. The wrapper is pass-through for the
forecasts themselves.
"""

from __future__ import annotations
import math

_EPS = 3e-12
_ITMAX = 300


# ---------------------------------------------------------------------------
# Chi-square tail probability via the regularized incomplete gamma function.
# Standard series / continued-fraction split (Abramowitz & Stegun 6.5).
# ---------------------------------------------------------------------------

def _gser(a: float, x: float) -> float:
    """Series for the regularized lower incomplete gamma P(a, x), x < a+1."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_ITMAX):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Continued fraction (modified Lentz) for the regularized upper
    incomplete gamma Q(a, x), x >= a+1."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, dof: float) -> float:
    """Survival function P(X > x) of a chi-square with ``dof`` degrees of
    freedom (fractional dof allowed): Q(dof/2, x/2)."""
    assert dof > 0
    if x <= 0.0:
        return 1.0
    a = 0.5 * dof
    xx = 0.5 * x
    if xx < a + 1.0:
        return 1.0 - _gser(a, xx)
    return _gcf(a, xx)


def chi2_ppf(p: float, dof: float, tol: float = 1e-10) -> float:
    """Quantile of the chi-square via bisection on the survival function."""
    assert 0.0 < p < 1.0
    lo = 0.0
    hi = dof + 40.0 * math.sqrt(2.0 * dof) + 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - chi2_sf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _gpd_fit(exc: list) -> tuple:
    """Probability-weighted-moments GPD fit (Hosking & Wallis 1987).

    Not method of moments: MOM needs finite variance (shape < 1/2) and on the
    heavy excess tails that d2 actually produces (Hill estimates ~0.7) it
    understates the shape by half and the deep tail by orders of magnitude —
    which is precisely the miscalibration the POT layer exists to remove.
    PWM is valid for shape < 1: a0 = mean, a1 = E[X (1-F(X))] from the order
    statistics; shape = 2 - a0/(a0 - 2 a1), sigma = 2 a0 a1/(a0 - 2 a1).
    """
    x = sorted(exc)
    n = len(x)
    a0 = sum(x) / n
    a1 = sum(((n - 1 - i) / (n - 1)) * x[i] for i in range(n)) / n
    denom = a0 - 2.0 * a1
    if denom <= 1e-12:                    # heavier than shape ~1: cap
        return 0.95, max(a0, 1e-12)
    gamma = min(max(2.0 - a0 / denom, -0.5), 0.95)
    sigma = max(2.0 * a0 * a1 / denom, 1e-12)
    return gamma, sigma


def _gpd_sf(e: float, gamma: float, sigma: float) -> float:
    """GPD survival function at excess e >= 0."""
    if abs(gamma) < 1e-9:
        return math.exp(-min(e / sigma, 700.0))
    arg = 1.0 + gamma * e / sigma
    if arg <= 0.0:                      # beyond the (finite) support
        return 0.0
    return arg ** (-1.0 / gamma)


# ---------------------------------------------------------------------------
# Small dense linear algebra on flat row-major k x k matrices (k is small).
# ---------------------------------------------------------------------------

def _cholesky(A: list, n: int, jitter: float = 1e-12) -> list:
    """Lower Cholesky factor of a (near-)PD flat row-major matrix."""
    L = [0.0] * (n * n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i * n + j]
            for t in range(j):
                s -= L[i * n + t] * L[j * n + t]
            if i == j:
                L[i * n + i] = math.sqrt(s) if s > jitter else math.sqrt(jitter)
            else:
                L[i * n + j] = s / L[j * n + j]
    return L


def _mahal2(L: list, v: list, n: int) -> float:
    """||L^{-1} v||^2 by forward substitution: v' (L L')^{-1} v."""
    w = [0.0] * n
    d2 = 0.0
    for i in range(n):
        s = v[i]
        for t in range(i):
            s -= L[i * n + t] * w[t]
        wi = s / L[i * n + i]
        w[i] = wi
        d2 += wi * wi
    return d2


def _top_eig(S: list, n: int, iters: int = 60) -> tuple:
    """Leading eigenpair of a symmetric flat matrix by power iteration.

    Deterministic start (uniform with a small index tilt to break ties) so the
    whole detector stays reproducible and portable to the JS twin.
    """
    v = [1.0 + 1e-3 * i for i in range(n)]
    norm = math.sqrt(sum(x * x for x in v))
    v = [x / norm for x in v]
    lam = 0.0
    for _ in range(iters):
        w = [sum(S[i * n + j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in w))
        if norm <= 0.0:
            return 0.0, v
        w = [x / norm for x in w]
        lam = sum(w[i] * sum(S[i * n + j] * w[j] for j in range(n))
                  for i in range(n))
        v = w
    return max(lam, 0.0), v


def _top_factors(S: list, n: int, r: int) -> list:
    """Up to ``r`` leading eigenpairs by power iteration with deflation.

    Stops early once an eigenvalue drops below 1% of the mean diagonal —
    factors below that carry no usable structure.
    """
    work = list(S)
    mean_diag = sum(S[i * n + i] for i in range(n)) / n
    out = []
    for _ in range(r):
        lam, v = _top_eig(work, n)
        if lam <= 0.01 * mean_diag:
            break
        out.append((lam, v))
        for i in range(n):                      # deflate: S -= lam * v v'
            for j in range(n):
                work[i * n + j] -= lam * v[i] * v[j]
    return out


def _solve_sym(A: list, b: list, n: int) -> list:
    """Solve A x = b for small symmetric positive-definite A (flat, n<=r)."""
    L = _cholesky(A, n)
    y = [0.0] * n
    for i in range(n):                          # forward: L y = b
        s = b[i]
        for t in range(i):
            s -= L[i * n + t] * y[t]
        y[i] = s / L[i * n + i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):              # back: L' x = y
        s = y[i]
        for t in range(i + 1, n):
            s -= L[t * n + i] * x[t]
        x[i] = s / L[i * n + i]
    return x


# ---------------------------------------------------------------------------
# The wrapper skater.
# ---------------------------------------------------------------------------

def mahalanobis(base, k: int, alpha: float = 0.02, scatter: str = "factor",
                factors: int = 1, shrink: float = 0.05, dfloor: float = 1e-3,
                guard_p: float = 0.99, adapt_after: int = 10,
                pot_level: float = 0.98, min_exc: int = 30):
    """Wrap a parade-wrapped skater with a streaming Mahalanobis anomaly score.

    Args:
        base: a skater whose state exposes ``state["z"]`` — i.e. already
            wrapped by :func:`skaters.parade` (as :func:`skaters.laplace` is).
        k: forecast horizon (must match base).
        alpha: EWMA rate for the location/scatter of z (memory ~ 1/alpha).
        scatter: how the near-singular scatter is tamed at scoring time.
            ``"factor"`` (default) fits one factor plus a diagonal,
            ``Sigma ~ lam * vv' + D`` (leading eigenpair by power iteration,
            residual per-horizon variances on the diagonal, exact inverse by
            Sherman--Morrison). The z-vector's structure is *known* — one
            dominant "all horizons surprised together" direction — so this
            models the degeneracy instead of flooring it, preserving
            sensitivity along the small "the forecasts disagreed" directions
            that identity shrinkage suppresses. ``"shrink"`` uses plain
            shrinkage toward the identity (the calibrated Ledoit--Wolf-style
            target, since the parade standardises each margin).
        factors: number of factors in the factor model (``scatter="factor"``).
            1 suffices for a single parade z-vector; a feature bank
            (:func:`zbank`) concatenating several engines' surprises has a
            few shared modes — use 2-4. Extraction deflates (power-iterate,
            subtract, repeat) and stops early below 1% of the mean diagonal;
            the inverse is exact Woodbury with an r x r solve.
        shrink: identity-shrinkage weight (``scatter="shrink"`` only).
        dfloor: floor on the factor model's residual variances, relative to
            the mean diagonal of the scatter (``scatter="factor"`` only).
            Bounds the false-alarm amplification from underestimated
            idiosyncratic variances.
        guard_p: chi-square level above which a tick's update is Huberised
            (downweighted by q/d2). Robustness against masking.
        adapt_after: consecutive guarded ticks after which the run is treated
            as a changepoint and updates resume at full weight.
        pot_level: the empirical-null quantile above which the tail is
            modelled by a streaming GPD (peaks over threshold). The
            Satterthwaite null is a two-moment bulk fit; extrapolating it to
            the 1e-4 tail overstates alarms by an order of magnitude on real
            data (the calibration panel measures this). Pickands--Balkema--%
            de Haan says exceedances are GPD whatever the law, so the tail
            gets its own fit — DSPOT's theorem on our multivariate statistic.
            (Lowering it to 0.95 improves sparse-anomaly calibration but
            regresses masking resistance under dense outliers — the
            exceedance set is contaminated and the GPD fit corrupted; see
            benchmarks/RESULTS.md §4c. 0.98 is kept for that reason.)
        min_exc: exceedances required before the GPD tail is trusted
            (bulk fit is used below that).

    After ``dists, state = f(y, state)``:
        state["d2"]     Mahalanobis distance of this tick's parade z-vector
                        (None until all k horizons have matured);
        state["pvalue"] its chi^2_k tail probability — calibrated, so under a
                        well-specified forecaster ~Uniform(0,1);
        state["run"]    consecutive ticks above the guard level (1 = isolated
                        spike ~ point outlier; growing ~ changepoint).
    """
    assert k >= 1
    assert 0.0 < alpha < 1.0
    assert 0.0 <= shrink < 1.0
    assert scatter in ("factor", "shrink")
    assert factors >= 1
    assert dfloor > 0.0
    assert min_exc >= 2, "min_exc < 2 divides by zero in the GPD fit"
    assert 0.0 < pot_level < 1.0
    assert 0.0 < guard_p < 1.0

    def _skater(y: float, state: dict | None):
        if state is None:
            state = {
                "k": k,
                "base": None,
                "mu": [0.0] * k,                     # calibrated prior mean
                "S": [1.0 if i == j else 0.0         # calibrated prior scatter
                      for i in range(k) for j in range(k)],
                "m2": float(k),                      # empirical-null mean of d2
                "v2": float(2 * k),                  # ...and variance (chi2_k prior)
                "exc": [],                           # POT excesses of d2
                "zeta": 1.0 - pot_level,             # P(d2 > t_pot), tracked
                # deep-evidence channel: -logpdf of y under the 1-step
                # predictive issued last tick. The parade clamps |z| at
                # ~7.03 (state sanity), so d2 SATURATES: a 250-sigma event
                # and a 10-sigma event become indistinguishable. nlp is
                # unbounded and restores resolution at depth; it gets its
                # own POT tail and joins by Bonferroni.
                "pend1": None, "nm": 0.0, "nv": 1.0, "n_exc": [],
                "n_zeta": 1.0 - pot_level, "n_n": 0,
                "run": 0, "d2": None, "pvalue": None,
                "skipped": 0, "last_dists": None,
            }
        elif state.get("k", k) != k:
            raise ValueError(
                f"state was built for k={state['k']} but this head has k={k};"
                " states are not interchangeable across k")
        # Harden the gate: a non-finite tick must not reach the body — a single
        # NaN poisons (or crashes) any forecaster's state permanently. Hold the
        # last forecasts, emit no score, count the skip, and carry on.
        if not (isinstance(y, (int, float)) and math.isfinite(y)):
            state["skipped"] += 1
            state["d2"] = None
            state["pvalue"] = None
            if state["last_dists"] is not None:
                return state["last_dists"], state
            from skaters.dist import Dist            # never-seen-data fallback
            return [Dist.gaussian(0.0, 1.0)] * k, state
        nlp = None
        if state["pend1"] is not None:
            lp = state["pend1"].logpdf(y)
            nlp = -lp if math.isfinite(lp) else 1e6
        # Winsorize absurd finite magnitudes before the body: double
        # arithmetic dies long before the float range ends (skaters' AR
        # raises OverflowError on a 1e300 tick and goes NaN by 1e100).
        # The parade z saturates at |z| = 7.03 anyway, so clamping at a
        # million sigmas is decision-invariant for every score this head
        # emits, and the deep-evidence channel above already saw raw y.
        y = min(max(y, -1e60), 1e60)
        if state["pend1"] is not None:
            m0, s0 = state["pend1"].mean, state["pend1"].std
            if math.isfinite(m0) and math.isfinite(s0):
                # magnitude-relative, NOT sigma-relative: after a
                # degenerate-variance stretch a legitimate value can sit
                # billions of sigmas out and must pass. Twelve orders
                # above the current level is unreachable by data, far
                # below the ~1e77 jump ratio where doubles actually die.
                w = 1e12 * (1.0 + abs(m0) + s0)
                y = min(max(y, m0 - w), m0 + w)
        dists, state["base"] = base(y, state["base"])
        state["last_dists"] = dists
        state["pend1"] = dists[0]
        bs = state["base"]
        z = bs.get("z") if isinstance(bs, dict) else None
        assert z is not None and len(z) == k, (
            "mahalanobis requires a parade-wrapped skater exposing "
            "state['z'] of length k (e.g. skaters.laplace)")

        # warmup / degenerate tick — and, for custom bases without the
        # parade clamp, a non-finite z, which would otherwise poison the
        # null moments permanently (inf -> m2/v2 -> nan nu next tick)
        if any(v is None or not math.isfinite(v) for v in z):
            state["d2"] = None
            state["pvalue"] = None
            return dists, state

        mu, S = state["mu"], state["S"]

        # Score under the CURRENT estimates — the point must not defend itself.
        v = [z[i] - mu[i] for i in range(k)]
        if scatter == "factor":
            # r factors + diagonal: Sigma ~ sum_j lam_j w_j w_j' + D, inverted
            # exactly by Woodbury (r x r solve). D keeps the *estimated*
            # residual variances, so disagreement directions retain their
            # true (small) scale.
            fac = _top_factors(S, k, factors)
            mean_diag = sum(S[i * k + i] for i in range(k)) / k
            floor = max(dfloor * mean_diag, 1e-12)
            D = [max(S[i * k + i]
                     - sum(lam * w[i] * w[i] for lam, w in fac), floor)
                 for i in range(k)]
            q1 = sum(v[i] * v[i] / D[i] for i in range(k))
            if not fac:
                d2 = q1
            else:
                r = len(fac)
                # b_j = w_j' D^{-1} v ;  B = diag(1/lam) + W' D^{-1} W
                b = [sum(w[i] * v[i] / D[i] for i in range(k))
                     for _, w in fac]
                B = [0.0] * (r * r)
                for a_ in range(r):
                    B[a_ * r + a_] = 1.0 / fac[a_][0]
                    for c_ in range(a_, r):
                        g = sum(fac[a_][1][i] * fac[c_][1][i] / D[i]
                                for i in range(k))
                        B[a_ * r + c_] += g
                        if c_ != a_:
                            B[c_ * r + a_] += g
                x = _solve_sym(B, b, r)
                d2 = q1 - sum(b[j] * x[j] for j in range(r))
        else:
            Ssh = [(1.0 - shrink) * S[i * k + j] + (shrink if i == j else 0.0)
                   for i in range(k) for j in range(k)]
            d2 = _mahal2(_cholesky(Ssh, k), v, k)

        # Empirical null, two layers. Bulk: two-moment Satterthwaite,
        # d2 ~ c * chi2_nu, seeded at the exact chi2_k moments. Tail: the
        # bulk fit extrapolates badly past its moments (measured: ~40x the
        # nominal false-alarm rate at 1e-4), so excesses over the null's
        # pot_level quantile get a streaming GPD (POT), and tail p-values
        # come from it: p = zeta * GPD_sf(d2 - t).
        m2 = max(state["m2"], 1e-9)
        v2 = max(state["v2"], 1e-9)
        c = max(v2 / (2.0 * m2), 1e-9)
        nu = min(max(2.0 * m2 * m2 / v2, 0.5), 1000.0)
        state["d2"] = d2
        t_pot = c * chi2_ppf(pot_level, nu)
        exc = state["exc"]
        t_scale = max(t_pot, 1e-9)
        if d2 > t_pot and len(exc) >= min_exc:
            # The GPD is authoritative in its region — the bulk chi2 tail is
            # exactly what it corrects (it understates p out here), so no
            # clamping against it. Excesses are threshold-RELATIVE: the null
            # (and hence t_pot) drifts, and absolute excesses pooled across
            # its history read as spurious tail weight.
            gamma, sigma = _gpd_fit(exc)
            state["pvalue"] = min(
                max(state["zeta"], 1e-12)
                * _gpd_sf((d2 - t_pot) / t_scale, gamma, sigma),
                1.0)
        else:
            state["pvalue"] = chi2_sf(d2 / c, nu)

        # nlp channel: speaks ONLY in its own POT tail — it exists for the
        # saturation regime (the z-clamp caps d2; nlp is unbounded), and in
        # the bulk it stays silent so the mahal p-value's null uniformity is
        # untouched. Combination is Bonferroni-style: min(p, 2 p_n).
        if nlp is not None and state["n_n"] >= min_exc:
            ns = math.sqrt(max(state["nv"], 1e-12))
            t_n = state["nm"] + 2.33 * ns
            if nlp > t_n and len(state["n_exc"]) >= min_exc:
                g2, s2 = _gpd_fit(state["n_exc"])
                p_n = max(state["n_zeta"], 1e-12) * _gpd_sf(
                    (nlp - t_n) / max(t_n - state["nm"], 1e-9), g2, s2)
                state["pvalue"] = min(state["pvalue"], 2.0 * p_n)

        # Huberised update with a changepoint escape hatch. The guard level is
        # the empirical null's quantile, so it tracks the learned calibration.
        q_guard = c * chi2_ppf(guard_p, nu)
        if d2 > q_guard:
            state["run"] += 1
            w = 1.0 if state["run"] > adapt_after else q_guard / d2
        else:
            state["run"] = 0
            w = 1.0
        a = alpha * w
        # Null moments update by WINSORIZATION, not downweighting: the Huber
        # weight bounds a linear update but the variance update is quadratic
        # in d2 (a * dm^2 ~ alpha * q * d2 still grows with the outlier), so a
        # downweighted outlier would widen its own null through v2. Clipping
        # at the guard bounds the influence outright; on the changepoint
        # escape (w == 1 with run exhausted) the raw d2 passes through so a
        # genuinely new regime can widen the null.
        d2n = d2 if w == 1.0 else min(d2, q_guard)
        dm = d2n - state["m2"]
        state["m2"] += alpha * dm
        state["v2"] = (1.0 - alpha) * state["v2"] + alpha * dm * (d2n - state["m2"])
        # POT layer maintenance: exceedance rate (EWMA of the indicator, at
        # the Huberised weight so anomaly runs cannot inflate their own tail)
        # and the excess buffer (clipped — one monster excess must not own
        # the moment fit; capped FIFO).
        aw = alpha * w
        state["zeta"] = (1.0 - aw) * state["zeta"] + aw * (1.0 if d2 > t_pot else 0.0)
        if d2 > t_pot:
            exc.append(min((d2 - t_pot) / t_scale, 50.0))
            if len(exc) > 250:
                exc.pop(0)
        if nlp is not None:
            state["n_n"] += 1
            ns = math.sqrt(max(state["nv"], 1e-12))
            t_n = state["nm"] + 2.33 * ns
            nw = min(nlp, state["nm"] + 6.0 * ns)   # winsorised moments
            dn = nw - state["nm"]
            state["nm"] += alpha * dn
            state["nv"] = (1.0 - alpha) * state["nv"] + alpha * dn * (nw - state["nm"])
            state["n_zeta"] = ((1.0 - aw) * state["n_zeta"]
                               + aw * (1.0 if nlp > t_n else 0.0))
            if nlp > t_n:
                state["n_exc"].append(
                    min((nlp - t_n) / max(t_n - state["nm"], 1e-9), 50.0))
                if len(state["n_exc"]) > 250:
                    state["n_exc"].pop(0)
        delta = v                                     # z - mu (pre-update)
        for i in range(k):
            mu[i] += a * delta[i]
        delta2 = [z[i] - mu[i] for i in range(k)]     # z - mu (post-update)
        for i in range(k):
            for j in range(k):
                S[i * k + j] = (1.0 - a) * S[i * k + j] + a * delta[i] * delta2[j]

        return dists, state

    _skater.__name__ = f"mahalanobis({getattr(base, '__name__', '?')}, k={k})"
    return _skater


def zbank(k: int = 3, sigmas: tuple = (0.03, 0.003),
          strides: tuple = (1, 4, 16), engine=None):
    """A bank of forecasters whose concatenated parade surprises form one z.

    The package's recurring move — a geometric grid over an unknown scale,
    mixed online — applied to detection: engines at every (residual-memory
    sigma) x (clock stride) gridpoint each maintain their own parade, and the
    bank exposes ``state["z"]`` as the concatenation (dimension
    ``len(sigmas) * len(strides) * k``). Feed it to :func:`mahalanobis` with
    ``k`` equal to that dimension and ``factors=2..4``: the bank's engines are
    highly correlated views of one stream, which is exactly the factor
    structure the scatter models.

    Strides reuse the :func:`skaters.multiscale` phase trick: stride s keeps s
    phase-shifted engine copies and each tick advances exactly the copy whose
    clock ends now, so every stride contributes a *fresh* surprise at every
    tick (no staleness, no detection delay) at an amortised cost of one engine
    step per (sigma, stride) per tick. A slow-memory engine spots anomalies
    against a long-held notion of normal; a coarse-clock engine spots drifts
    a fine clock absorbs.

    Forecasts pass through from the (first sigma, stride 1) engine.

    Args:
        k: per-engine horizon (also the finest engine's output length).
        sigmas: ``scale_alpha`` grid for the default laplace engine.
        strides: decimation grid; must include 1 (the pass-through engine).
        engine: optional factory ``engine(sigma) -> skater`` overriding the
            default ``laplace(k, scale_alpha=sigma)``. The returned skater
            must expose parade state (``state["z"]``), as laplace does.
    """
    assert 1 in strides, "strides must include 1 (the pass-through engine)"
    from skaters import laplace as _laplace
    make = engine if engine is not None else (
        lambda sig: _laplace(k, scale_alpha=sig))
    fs = {sig: make(sig) for sig in sigmas}      # skaters are pure: one per sigma

    def _skater(y: float, state: dict | None):
        if state is None:
            state = {"t": 0,
                     "sub": {(sig, s): [None] * s
                             for sig in sigmas for s in strides},
                     "z": None}
        t = state["t"]
        z = []
        dists_out = None
        for sig in sigmas:
            for s in strides:
                copies = state["sub"][(sig, s)]
                ph = t % s
                dists, copies[ph] = fs[sig](y, copies[ph])
                if sig == sigmas[0] and s == 1:
                    dists_out = dists              # pass-through forecasts
                cz = copies[ph].get("z") if isinstance(copies[ph], dict) else None
                z.extend(cz if cz is not None else [None] * k)
        state["z"] = z
        state["t"] = t + 1
        return dists_out, state

    _skater.__name__ = (f"zbank(k={k}, sigmas={list(sigmas)}, "
                        f"strides={list(strides)})")
    return _skater
