"""Forward support search with conditional quadratic curvature and QP checks.

The Schur score is a local ranking approximation. Each proposed addition is
evaluated by the original bounded QP with the linear stale-hops constraint.
The algorithm is a greedy cardinality heuristic, not a global L0 optimizer.
"""
import time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def forward_supports(h, b, a, lo, hi, mandatory=(), budgets=(3,5,10,20), lookahead=6):
    p=len(b)
    support=list(map(int,mandatory))
    delta=np.zeros(p)
    records=[]

    def solve(selected,warm):
        selected=np.asarray(selected,dtype=int)
        if not len(selected):
            return np.zeros(p),bool(np.all(lo<=1e-8) and np.all(hi>=-1e-8)),"empty support",0
        hs=h[np.ix_(selected,selected)];bs=b[selected]
        ans=minimize(lambda d:float(.5*d@hs@d-bs@d),np.clip(warm[selected],lo[selected],hi[selected]),
             jac=lambda d:hs@d-bs,method="SLSQP",bounds=Bounds(lo[selected],hi[selected]),
             constraints=[LinearConstraint(a[selected],-np.inf,0.)],
             options=dict(maxiter=300,ftol=1e-11))
        out=np.zeros(p);out[selected]=ans.x
        ok=bool(ans.success and np.max(lo-out)<=1e-8 and np.max(out-hi)<=1e-8 and a@out<=1e-8)
        return out,ok,str(ans.message),int(ans.nit)

    delta,feasible,message,nit=solve(support,delta)
    # A required missing-defect correction can itself violate the sensory
    # halfspace; add a compensating coordinate before optimizing the budget.
    while not feasible and len(support)<min(budgets):
        remaining=np.array([j for j in range(p) if j not in support],dtype=int)
        leverage=np.minimum(a[remaining]*lo[remaining],a[remaining]*hi[remaining])
        candidates=remaining[np.argsort(leverage,kind="stable")[:lookahead]]
        evaluated=[]
        for j in candidates:
            dd,ok,msg,nn=solve(support+[int(j)],delta)
            evaluated.append((float(.5*dd@h@dd-b@dd) if ok else np.inf,int(j),dd,ok,msg,nn))
        win=min(evaluated,key=lambda t:t[0])
        if not np.isfinite(win[0]):
            chosen=int(candidates[0]);support.append(chosen)
            delta,feasible,message,nit=solve(support,delta)
        else:
            _,chosen,delta,feasible,message,nit=win;support.append(chosen)
        records.append(dict(step=len(support),selected_index=chosen,stage="required_feasibility_repair",objective_delta=float(.5*delta@h@delta-b@delta),feasible=feasible,evaluated_candidates=len(candidates),qp_iterations=nit))
    if not feasible:
        raise RuntimeError("Mandatory defect repair could not obtain a feasible initial sparse support")

    result={}
    if len(support) in budgets:
        result[len(support)]=np.array(support,dtype=int)
    while len(support)<max(budgets):
        remaining=np.array([j for j in range(p) if j not in support],dtype=int)
        residual=b-h@delta
        ss=np.array(support,dtype=int)
        free=ss[(delta[ss]>lo[ss]+1e-7)&(delta[ss]<hi[ss]-1e-7)] if len(ss) else ss
        curvature=np.diag(h)[remaining].copy()
        lagrange=0.
        active=bool(abs(a@delta)<1e-7)
        free_constraint=False
        if len(free):
            inv=np.linalg.inv(h[np.ix_(free,free)])
            cross=h[np.ix_(remaining,free)]
            curvature-=np.einsum("ij,jk,ik->i",cross,inv,cross)
            if active:
                denom=float(a[free]@inv@a[free])
                if denom>1e-14:
                    free_constraint=True
                    lagrange=max(float(a[free]@residual[free]/(a[free]@a[free])),0.)
                    effective=a[remaining]-cross@inv@a[free]
                    curvature+=effective**2/denom
        curvature=np.maximum(curvature,1e-10)
        slope=residual[remaining]-lagrange*a[remaining]
        step=np.clip(slope/curvature,lo[remaining],hi[remaining])
        if active and not free_constraint:
            # No existing free direction can offset the active halfspace.
            step=np.where(a[remaining]*step>0.,0.,step)
        benefit=slope*step-.5*curvature*step**2
        trial_order=remaining[np.argsort(-benefit,kind="stable")[:lookahead]]
        evaluated=[]
        started=time.perf_counter()
        for j in trial_order:
            dd,ok,msg,nn=solve(support+[int(j)],delta)
            objective=float(.5*dd@h@dd-b@dd) if ok else np.inf
            evaluated.append((objective,int(j),dd,ok,msg,nn))
        win=min(evaluated,key=lambda t:t[0])
        if not np.isfinite(win[0]):
            raise RuntimeError("Nested support lost the feasible previous solution")
        objective,chosen,delta,feasible,message,nit=win
        support.append(chosen)
        records.append(dict(step=len(support),selected_index=chosen,stage="schur_forward",objective_delta=objective,
                            feasible=feasible,evaluated_candidates=len(trial_order),qp_iterations=nit,
                            seconds=time.perf_counter()-started,stale_active=active,
                            approximate_lagrange_multiplier=lagrange))
        if len(support) in budgets:
            result[len(support)]=np.array(support,dtype=int)
    return result,records
