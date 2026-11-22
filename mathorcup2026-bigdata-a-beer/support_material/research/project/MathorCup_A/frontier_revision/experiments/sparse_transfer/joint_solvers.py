"""Convex geometry aware support search and mixed-integer certificates.

SCIP solves a mixed integer convex quadratically constrained relaxation of the
FULL nonlinear sensory-checked problem. A feasible decoded candidate supplies
an upper bound only after the independent frozen predictor check. A solver dual
bound supplies a lower bound. The remaining gap is always explicitly reported.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]='1'
import time
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds,LinearConstraint,minimize
import osqp
from pyscipopt import Model,quicksum


def sparse_qp(problem,beer,support=None,margin=0.,initial=None,penalty=0.,weights=None,eps=1e-8):
    """Convex QP plus weighted L1 epigraph, with optional frozen support."""
    p=problem.p;c=problem.contexts[beer]
    A,rhs=problem.linear_constraints(beer,margin=margin)
    lo,hi=c['lo'].copy(),c['hi'].copy()
    if support is not None:
        frozen=np.ones(p,bool);frozen[np.asarray(support,int)]=False
        lo[frozen]=0;hi[frozen]=0
    if np.any(lo>hi+1e-12):return None,dict(status='inconsistent bounds')
    if penalty>0:
        I=sparse.eye(p,format='csc');zero=sparse.csc_matrix((p,p))
        H=sparse.block_diag([sparse.csc_matrix(problem.H),zero],format='csc')
        q=np.r_[c['gradient'],penalty*(np.ones(p) if weights is None else weights)]
        AA=sparse.vstack([sparse.hstack([sparse.csc_matrix(A),sparse.csc_matrix((len(A),p))]),
                          sparse.hstack([I,zero]),sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csc')
        lower=np.r_[np.full(len(A),-np.inf),lo,np.full(2*p,-np.inf)]
        upper=np.r_[rhs,hi,np.zeros(2*p)]
    else:
        H=sparse.csc_matrix(problem.H);q=c['gradient']
        AA=sparse.vstack([sparse.csc_matrix(A),sparse.eye(p)],format='csc')
        lower=np.r_[np.full(len(A),-np.inf),lo];upper=np.r_[rhs,hi]
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(H,format='csc'),q=q,A=AA,l=lower,u=upper,
                 verbose=False,eps_abs=eps,eps_rel=eps,max_iter=30000,polishing=True,adaptive_rho=True)
    if initial is not None:
        solver.warm_start(x=np.r_[initial,np.abs(initial)] if penalty>0 else initial)
    t0=time.perf_counter();res=solver.solve()
    info=dict(status=res.info.status,iterations=int(res.info.iter),seconds=time.perf_counter()-t0)
    if res.info.status_val not in (1,2):return None,info
    d=res.x[:p];v=float(max(np.max(A@d-rhs),np.max(lo-d),np.max(d-hi)))
    info.update(violation=v,objective=problem.objective(beer,d))
    if v>max(1e-6,10*eps):return None,info
    return d,info


def reweighted_candidates(problem,beer,margin=0.,lambdas=(.01,.03,.1,.3,1.,3.,10.),epsilon=.1,steps=5):
    dense,info=sparse_qp(problem,beer,margin=margin)
    if dense is None:return [],[dict(stage='dense',**info)]
    candidates=[];records=[];seen=set()
    for lam in lambdas:
        d=dense.copy()
        for it in range(steps):
            weights=1/(np.abs(d)+epsilon)
            d1,info=sparse_qp(problem,beer,margin=margin,initial=d,penalty=lam,weights=weights)
            records.append(dict(stage='reweighted_l1',lambda_=lam,iteration=it,margin=margin,**info))
            if d1 is None:break
            d=d1
            support=np.union1d(np.flatnonzero(np.abs(d)>1e-5),problem.contexts[beer]['mandatory']).astype(int)
            skey=tuple(support.tolist())
            if skey in seen:continue
            seen.add(skey)
            polished,inf=sparse_qp(problem,beer,support,margin,d)
            if polished is not None:
                # SLSQP refines numerical feasibility after selecting a support.
                q,dd,violation=problem.polish(beer,support,margin,polished)
                if violation<1e-8:
                    candidates.append(dict(delta=dd,support=support,margin=margin,lambda_=lam,iteration=it,
                                           method='reweighted_l1',objective=problem.objective(beer,dd)))
    return candidates,records


def solve_cardinality(problem,beer,rho=.10,margin=0.,time_limit=120.,warm=None,logfile=None,seed=20261024):
    """min ||delta||_0 subject to distance reduction >= rho and linear guards.

    v = chol(P)^T (z0 + delta - mu). The one nonlinear constraint is the
    convex Euclidean ball ||v||² <= (1-rho)² ||v0||². Binary links use exact
    training box bounds, with no arbitrary large constants.
    """
    c=problem.contexts[beer];p=problem.p
    m=Model(f'joint_sparse_{beer}_{problem.policy}')
    m.setRealParam('limits/time',float(time_limit));m.setRealParam('limits/gap',0.)
    m.setRealParam('numerics/feastol',1e-8);m.setIntParam('randomization/randomseedshift',int(seed))
    m.setIntParam('display/verblevel',4 if logfile else 0)
    m.setIntParam('parallel/maxnthreads',1)
    m.setBoolParam('constraints/nonlinear/assumeconvex',True)
    if logfile:m.setLogfile(str(logfile))
    d=[m.addVar(f'd{j}',lb=min(0,c['lo'][j]),ub=max(0,c['hi'][j])) for j in range(p)]
    b=[m.addVar(f'b{j}',vtype='B',lb=1 if j in c['mandatory'] else 0) for j in range(p)]
    for j in range(p):
        m.addCons(d[j]<=float(c['hi'][j])*b[j]);m.addCons(d[j]>=float(c['lo'][j])*b[j])
    A,rhs=problem.linear_constraints(beer,margin=margin)
    for row,rr in zip(A,rhs):m.addCons(quicksum(float(row[j])*d[j] for j in range(p))<=float(rr))
    L=np.linalg.cholesky(problem.P).T
    v0=L@(c['z0']-problem.mu)
    # Linear factorization makes the positive semidefinite geometry explicit.
    v=[m.addVar(f'v{j}',lb=None,ub=None) for j in range(p)]
    for i in range(p):
        m.addCons(v[i]==float(v0[i])+quicksum(float(L[i,j])*d[j] for j in range(p) if abs(L[i,j])>1e-15))
    radius2=(1-rho)**2*float(v0@v0)
    m.addCons(quicksum(vj*vj for vj in v)<=radius2,name='target_distance_ball')
    m.setObjective(quicksum(b),'minimize')
    warm_added=False
    if warm is not None:
        dd=np.asarray(warm);vv=v0+L@dd
        if vv@vv<=radius2+1e-7 and np.max(A@dd-rhs)<=1e-7:
            sol=m.createSol()
            for j in range(p):
                m.setSolVal(sol,d[j],float(dd[j]));m.setSolVal(sol,b[j],float(abs(dd[j])>1e-7 or j in c['mandatory']))
                m.setSolVal(sol,v[j],float(vv[j]))
            warm_added=bool(m.addSol(sol))
    t0=time.perf_counter();m.optimize();elapsed=time.perf_counter()-t0
    out=[]
    for sol in m.getSols():
        dd=np.array([m.getSolVal(sol,x) for x in d]);ss=np.array([j for j,x in enumerate(b) if m.getSolVal(sol,x)>.5])
        out.append(dict(delta=dd,support=ss,method='SCIP_MIQCP',margin=margin,rho=rho))
    record=dict(beer=int(beer),policy=problem.policy,rho=rho,margin=margin,status=str(m.getStatus()),
                dual_bound=float(m.getDualbound()),primal_bound=float(m.getPrimalbound()),gap=float(m.getGap()),
                nodes=int(m.getNNodes()),seconds=elapsed,solutions=len(out),warm_added=warm_added,
                solver='SCIP 10.0.2 / PySCIPOpt 6.3.0',style_epsilon=problem.style_epsilon)
    m.freeProb()
    return out,record


def prune_support(problem,beer,delta,support,margin=0.,rho=.1,max_trials=12):
    """Feasible backward refinement; every accepted deletion passes the full model.

    The Schur diagonal gives an inexpensive *ranking heuristic*, never a bound
    under the active constraints. Trials solve the full fixed-support problem.
    No failed trial is used as a global support infeasibility claim.
    """
    s=np.asarray(sorted(support),int);d=np.asarray(delta).copy();records=[]
    mandatory=set(problem.contexts[beer]['mandatory'])
    while len(s)>len(mandatory):
        H=problem.H[np.ix_(s,s)]
        invdiag=np.diag(np.linalg.inv(H))
        costs=d[s]**2/np.maximum(2*invdiag,1e-12)
        order=[int(s[j]) for j in np.argsort(costs) if int(s[j]) not in mandatory][:max_trials]
        proposals=[];payload=[]
        for drop in order:
            ss=s[s!=drop]
            dd,info=sparse_qp(problem,beer,ss,margin,d)
            if dd is None:
                records.append(dict(drop=drop,size=len(ss),qp_feasible=False));continue
            q,dd,v=problem.polish(beer,ss,margin,dd,rho=rho)
            if v>1e-8:
                records.append(dict(drop=drop,size=len(ss),qp_feasible=False));continue
            row,z=problem.decode(beer,dd,ss);row.update(required_rho=rho,required_proxy_margin=margin)
            proposals.append(row);payload.append((drop,ss,dd))
        if not proposals:break
        checks,_=problem.check_rows(proposals)
        for i,(drop,ss,dd) in enumerate(payload):
            records.append(dict(drop=drop,size=len(ss),qp_feasible=True,accepted=bool(checks.iloc[i].accepted),
                                distance_reduction=float(checks.iloc[i].target_distance_reduction)))
        valid=checks[checks.accepted]
        if not len(valid):break
        ix=int(valid.sort_values('quadratic_objective').index[0]);_,s,d=payload[ix]
    return dict(delta=d,support=s,margin=margin,rho=rho,method='feasible_backward'),records
