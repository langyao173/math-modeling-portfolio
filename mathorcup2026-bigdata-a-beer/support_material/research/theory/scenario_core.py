"""Finite-scenario classifier guards and exact held-out binomial bounds.

These utilities do not turn model agreement into physical intervention success.
All inference is conditional on the fixed observed data and refitting law.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import beta, binom


def cp_lower(successes: int, trials: int, delta: float = .05) -> float:
    """One-sided (1-delta) Clopper-Pearson lower success-probability bound."""
    if not 0 <= successes <= trials or trials < 1 or not 0 < delta < 1:
        raise ValueError("Require 0 <= successes <= trials, trials >= 1, 0<delta<1")
    return 0. if successes == 0 else float(beta.ppf(delta, successes, trials-successes+1))


def cp_upper(successes: int, trials: int, delta: float = .05) -> float:
    """One-sided (1-delta) Clopper-Pearson upper success-probability bound."""
    if not 0 <= successes <= trials or trials < 1 or not 0 < delta < 1:
        raise ValueError("Invalid binomial arguments")
    return 1. if successes == trials else float(beta.ppf(1-delta, successes+1, trials-successes))


def zero_failure_trials(target_probability=.9, delta=.05, multiplicity=1):
    """Trials needed if every validation scenario succeeds, with Bonferroni."""
    if not 0 < target_probability < 1 or multiplicity < 1 or not 0 < delta < 1:
        raise ValueError("Invalid planning arguments")
    return math.ceil(math.log(delta/multiplicity) / math.log(target_probability))


def classification_scenario_matrix(weights, intercepts, z0, lower, upper, eta=0.):
    """Return A,rhs for A @ [d,v] <= rhs, with ONE v per joint scenario.

    weights[r,k,:] and intercepts[r,k] are the Tripel-minus-class-k
    logit differences. All classifiers share the same fixed coordinate map.
    lower <= d <= upper must contain all values admitted by support links;
    using min(0, original_lower), max(0, original_upper) is conservative.
    v[r]=0 enforces every class margin in refit r; v[r]=1 exempts it.
    The caller must reuse v[r] for that refit's exact sensory guard and impose
    sum(v)<=q. This routine does not encode the nonlinear sensory ensemble.
    """
    W=np.asarray(weights,float); a=np.asarray(intercepts,float)
    z0=np.asarray(z0,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    if W.ndim != 3 or a.shape != W.shape[:2] or z0.shape != (W.shape[2],):
        raise ValueError("Expected W=(R,K,p), a=(R,K), z0=(p,)")
    if lo.shape != z0.shape or hi.shape != z0.shape or np.any(lo>hi):
        raise ValueError("Invalid box")
    R,K,p=W.shape
    target=np.broadcast_to(np.asarray(eta,float),(R,K))
    base=np.einsum('rkp,p->rk',W,z0)+a
    minimum=base+np.sum(np.where(W>=0,W*lo,W*hi),axis=2)
    big_m=np.maximum(0.,target-minimum)
    A=np.zeros((R*K,p+R));A[:,:p]=-W.reshape(-1,p)
    for r in range(R):A[r*K:(r+1)*K,p+r]=-big_m[r]
    rhs=(base-target).ravel()
    return A,rhs,big_m


def paired_mean_lower(new_joint, old_joint, delta=.05):
    """Hoeffding lower bound for E[new_joint-old_joint] on paired iid scenarios.

    Both candidate identities must be fixed before these scenarios are seen.
    A difference takes values in [-1,1], hence the sqrt(2 log(1/delta)/N) term.
    """
    x=np.asarray(new_joint,float);y=np.asarray(old_joint,float)
    if x.shape != y.shape or x.ndim != 1 or not len(x):
        raise ValueError("Expected aligned nonempty one-dimensional outcomes")
    if np.any((x<0)|(x>1)|(y<0)|(y>1)) or not 0<delta<1:
        raise ValueError("Outcomes must lie in [0,1]")
    diff=x-y
    return float(diff.mean()-math.sqrt(2*math.log(1/delta)/len(diff)))


def residual_tolerance_bound(residual_maxima, alpha=.1, delta=.05):
    """One-sided nonparametric (1-alpha)-content, (1-delta)-confidence bound.

    Each calibration value must be the maximum h_r(c)-proxy(c) over the SAME
    finite candidate library frozen before these iid calibration scenarios.
    If N cannot support any finite order-statistic bound, return +infinity.
    This is a conditional model-law tolerance bound, not a physical guarantee.
    """
    values=np.asarray(residual_maxima,float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite one-dimensional calibration maxima")
    if not 0<alpha<1 or not 0<delta<1:
        raise ValueError("alpha and delta must lie in (0,1)")
    n=len(values)
    eligible=[k for k in range(1,n+1) if binom.sf(k-1,n,1-alpha)<=delta]
    if not eligible:return float('inf'),None
    k=eligible[0]
    return float(np.partition(values,k-1)[k-1]),k


if __name__ == '__main__':
    import json
    print(json.dumps({
        'N200_single_profile':{str(k):cp_lower(k,200) for k in [180,185,190,195,200]},
        'N200_simultaneous_11_profiles':{str(k):cp_lower(k,200,.05/11) for k in [180,185,190,195,200]},
        'zero_failures_to_certify_p90_all11':zero_failure_trials(.9,.05,11),
        'scope':'Conditional refit agreement under the chosen iid model-scenario law; not physical success'
    },indent=2))
