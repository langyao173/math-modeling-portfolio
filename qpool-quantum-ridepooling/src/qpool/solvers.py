from __future__ import annotations
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from .qubo import PackingInstance,packing_qubo


def packing_constraints(instance):
    n=len(instance.weights)
    # Clique rows strengthen the LP, but only when all implied edges really exist.
    from itertools import combinations
    edge_set=set(instance.edges);covered=set();sets=[]
    if instance.labels and all('vehicle' in d and 'requests' in d for d in instance.labels):
        groups={}
        for i,d in enumerate(instance.labels):
            for key in [('v',d['vehicle'])]+[('r',r) for r in d['requests']]:groups.setdefault(key,[]).append(i)
        for g in groups.values():
            pairs={tuple(sorted(e)) for e in combinations(g,2)}
            if len(g)>1 and pairs<=edge_set:sets.append(g);covered|=pairs
    sets += [list(e) for e in sorted(edge_set-covered)]
    rows=[];cols=[]
    for j,g in enumerate(sets):rows.extend([j]*len(g));cols.extend(g)
    A=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(sets),n)).tocsc()
    return A

def solve_milp(instance,time_limit=10.0):
    start=time.perf_counter();n=len(instance.weights)
    if n==0:return np.zeros(0,int),{'elapsed_s':0.,'optimal':True,'gap':0.,'bound':0.}
    A=packing_constraints(instance)
    res=milp(-instance.weights,integrality=np.ones(n),bounds=Bounds(0,1),
             constraints=LinearConstraint(A,-np.inf,1),
             options={'time_limit':time_limit,'mip_rel_gap':0.0})
    if res.x is None:
        x=solve_greedy(instance)[0];optimal=False
    else:x=np.rint(res.x).astype(int);optimal=res.status==0
    assert instance.evaluate(x)['feasible']
    bound=getattr(res,'mip_dual_bound',None)
    return x,{'elapsed_s':time.perf_counter()-start,'optimal':optimal,
              'gap':None if getattr(res,'mip_gap',None) is None else float(res.mip_gap),
              'bound':None if bound is None else -float(bound),'status':int(res.status)}

def solve_greedy(instance):
    st=time.perf_counter();adj=instance.adjacency();n=len(adj);best=np.zeros(n,int)
    for mode in ('weight','density','sqrt'):
        power={'weight':0,'density':1,'sqrt':.5}[mode]
        order=sorted(range(n),key=lambda i:(-instance.weights[i]/(1+len(adj[i]))**power,i))
        blocked=set();x=np.zeros(n,int)
        for i in order:
            if i not in blocked and instance.weights[i]>0:x[i]=1;blocked|=adj[i];blocked.add(i)
        if instance.weights@x>instance.weights@best:best=x
    return best,{'elapsed_s':time.perf_counter()-st,'optimal':False}

def solve_local(instance,time_limit=.25):
    st=time.perf_counter();x=solve_greedy(instance)[0];adj=instance.adjacency();w=instance.weights
    improved=True
    while improved and time.perf_counter()-st<time_limit:
        improved=False
        for u in np.argsort(-w):
            if x[u]:continue
            victims=[v for v in adj[u] if x[v]]
            if w[u]>sum(w[v] for v in victims)+1e-10:
                x[victims]=0;x[u]=1;improved=True
        if improved:continue
        inactive=[i for i in np.argsort(-w) if not x[i]]
        for k,u in enumerate(inactive):
            if time.perf_counter()-st>=time_limit:break
            for v in inactive[k+1:]:
                if v in adj[u]:continue
                victims=[j for j in adj[u]|adj[v] if x[j]]
                if w[u]+w[v]>sum(w[j] for j in victims)+1e-10:
                    x[victims]=0;x[u]=x[v]=1;improved=True;break
            if improved:break
    return x,{'elapsed_s':time.perf_counter()-st,'optimal':False}

def solve_sa(instance,seed=0,sweeps=300,restarts=8):
    """Local classical Metropolis SA, explicitly NOT Kaiwu SA or quantum output."""
    st=time.perf_counter();Q=packing_qubo(instance);n=len(Q)
    if n==0:return np.zeros(0,int),{'elapsed_s':0,'optimal':True}
    rng=np.random.default_rng(seed);linear=np.diag(Q);off=Q+Q.T-2*np.diag(linear)
    best=solve_greedy(instance)[0];best_e=float(best@Q@best)
    for restart in range(restarts):
        x=(np.zeros(n,dtype=int) if restart==0 else rng.integers(0,2,n))
        field=linear+off@x;e=float(x@Q@x)
        temps=np.geomspace(max(1,float(np.max(instance.weights))),.025,sweeps)
        for temp in temps:
            for i in rng.permutation(n):
                d=(1-2*x[i])*field[i]
                if d<=0 or rng.random()<np.exp(-min(d/temp,745)):
                    change=1-2*x[i];x[i]+=change;field+=change*off[:,i];e+=d
                    if e<best_e-1e-9:best=x.copy();best_e=e
    # All one-flip local minima are feasible under this packing penalty.
    changed=True
    while changed:
        changed=False
        for i in range(n):
            d=(1-2*best[i])*(linear[i]+off[i]@best)
            if d<-1e-10:best[i]=1-best[i];changed=True
    assert instance.evaluate(best)['feasible']
    return best,{'elapsed_s':time.perf_counter()-st,'optimal':False,'seed':seed,
                 'sweeps':sweeps,'restarts':restarts,'backend':'local_classical_metropolis'}

def tv_worst_probability(p,z,radius):
    """Exact minimum p'@z over TV(p',p)<=radius on a finite simplex."""
    p=np.asarray(p,float);z=np.asarray(z,float)
    if p.shape!=z.shape or np.any(p<0) or abs(p.sum()-1)>1e-8 or not 0<=radius<=1:
        raise ValueError('Invalid finite-support total-variation problem.')
    q=p.copy();lo=np.argsort(z,kind='stable');hi=lo[::-1];i=j=0;budget=float(radius)
    while budget>1e-12 and i<len(p) and j<len(p):
        a,b=int(lo[i]),int(hi[j])
        if z[a]>=z[b]-1e-14:break
        mass=min(1-q[a],q[b],budget)
        q[a]+=mass;q[b]-=mass;budget-=mass
        if q[a]>=1-1e-12:i+=1
        if q[b]<=1e-12:j+=1
    return q,float(q@z)

def solve_joint_tv(instance,features,p,radius=.1,future_weight=1.,time_limit=10.,max_iter=30):
    """Cutting-plane MILP for a *linear feature* joint-TV robust objective.

    This is not the true Bellman value, and is not mislabeled as a QUBO solve.
    It returns a feasible lower bound and the last valid master upper bound.
    """
    st=time.perf_counter();n=len(instance.weights);Z=np.asarray(features,float);p=np.asarray(p,float)
    if Z.shape!=(len(p),n):raise ValueError('features shape must be support_size x n.')
    if future_weight<0:raise ValueError('future_weight must be nonnegative.')
    if n==0:return np.zeros(0,int),{'elapsed_s':0,'optimal':True,'robust_value':0,'iterations':0}
    A=packing_constraints(instance);cuts=[p.copy()];best=np.zeros(n,int);best_val=0.;upper=np.inf
    from scipy.sparse import hstack,vstack,csr_matrix
    base=hstack([A,csr_matrix((A.shape[0],1))]).tocsr()
    M=max(1.,float(np.max(np.sum(np.abs(Z),axis=1))))
    optimal=False
    for it in range(max_iter):
        left=time_limit-(time.perf_counter()-st)
        if left<=0:break
        R=csr_matrix(np.vstack([np.r_[-q@Z,1.] for q in cuts]))
        C=vstack([base,R]).tocsc();ub=np.r_[np.ones(A.shape[0]),np.zeros(len(cuts))]
        res=milp(np.r_[-instance.weights,-future_weight],integrality=np.r_[np.ones(n),0],
                 bounds=Bounds(np.r_[np.zeros(n),-M],np.r_[np.ones(n),M]),
                 constraints=LinearConstraint(C,-np.inf,ub),options={'time_limit':left,'mip_rel_gap':0.})
        if res.x is None:break
        x=np.rint(res.x[:n]).astype(int);assert instance.evaluate(x)['feasible']
        q,worst=tv_worst_probability(p,Z@x,radius)
        val=float(instance.weights@x+future_weight*worst)
        if val>best_val:best=x;best_val=val
        bound=getattr(res,'mip_dual_bound',None)
        if bound is not None:upper=min(upper,-float(bound))
        if upper-best_val<=1e-7:optimal=True;break
        if any(np.allclose(q,old,atol=1e-12) for old in cuts):break
        cuts.append(q)
    return best,{'elapsed_s':time.perf_counter()-st,'optimal':optimal,
                 'robust_value':best_val,'robust_upper_bound':None if not np.isfinite(upper) else upper,
                 'iterations':len(cuts),'backend':'joint_tv_classical_cutting_plane'}
