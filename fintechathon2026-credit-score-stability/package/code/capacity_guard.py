"""Minimum-change temporal scores with exact operating-capacity membership.

Inputs are predictions, previous scores, contemporaneous gates and immutable
row IDs. Neither labels nor future observations are accepted by the API.
"""
import math
from numbers import Integral
import numpy as np

def order(probabilities,row_ids):
    return np.lexsort((np.asarray(row_ids,dtype=str),-np.asarray(probabilities,float)))

def protect_membership(base,target,row_ids,quotas):
    p=np.asarray(base,float);z=np.asarray(target,float);ids=np.asarray(row_ids,dtype=str)
    if p.ndim!=1 or p.shape!=z.shape or len(ids)!=len(p):raise ValueError('Vector shape mismatch')
    if len(set(ids))!=len(ids):raise ValueError('Row identifiers must be unique within batch')
    if not np.isfinite(p).all() or not np.isfinite(z).all() or (p<0).any() or (p>1).any() or (z<0).any() or (z>1).any():raise ValueError('Probabilities must be finite in [0,1]')
    n=len(p)
    if n == 0 or ids.ndim != 1 or any(not v.strip() for v in ids):raise ValueError('A nonempty identified batch is required')
    try:quotas=list(quotas)
    except TypeError:raise ValueError('Quotas must be a sequence of integers')
    if not quotas or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,Integral) or not 0<=k<=n for k in quotas):raise ValueError('Quotas must be integers between zero and batch size')
    cuts=sorted(set(int(k) for k in quotas if 0<k<n));o=order(p,ids);rank=np.empty(n,int);rank[o]=np.arange(n)
    # Resolve boundary ties in the same (probability, ID) order used by the list.
    # This correction is at most 2e-10 in probability and is only used for ties.
    boundary_tie=any(p[o[k-1]]==p[o[k]] for k in cuts)
    scale=(1-2e-10)*p+2e-10*(n-rank)/(n+1) if boundary_tie else p
    lo=np.zeros(n);hi=np.ones(n);thresholds=[]
    for k in cuts:
        threshold=(scale[o[k-1]]+scale[o[k]])/2
        low_selected=np.nextafter(threshold,1.)
        if low_selected<=threshold:raise ValueError('Boundary precision exhausted')
        lo[rank<k]=np.maximum(lo[rank<k],low_selected)
        hi[rank>=k]=np.minimum(hi[rank>=k],threshold)
        thresholds.append(dict(quota=k,threshold=float(threshold),selected_floor=float(low_selected)))
    if (lo>hi).any():raise ValueError('Inconsistent protected boundaries')
    q=np.clip(z,lo,hi)
    oq=order(q,ids)
    for k in cuts:
        if set(o[:k])!=set(oq[:k]):raise AssertionError('Membership protection failed')
    return q,dict(lower=lo,upper=hi,thresholds=thresholds,boundary_tie=boundary_tie,
                  guard_changed=np.abs(q-z)>1e-12)

def score_batch(base,previous,gate,row_ids,*,quota,radius=.01):
    p=np.asarray(base,float);old=np.asarray(previous,float);g=np.asarray(gate)
    if g.dtype != np.dtype(bool):raise ValueError('Gate must be boolean')
    if p.shape!=old.shape or p.shape!=g.shape:raise ValueError('State vector shape mismatch')
    if not 0<radius<1:raise ValueError('Radius outside (0,1)')
    if not np.isfinite(old).all() or (old<0).any() or (old>1).any():raise ValueError('Previous probabilities must be finite in [0,1]')
    z=np.where(g,np.clip(p,old-radius,old+radius),p)
    q,receipt=protect_membership(p,z,row_ids,[quota] if isinstance(quota,Integral) else quota)
    receipt.update(temporal_candidate=z,final_probability=q,gate=g,
        unavoidable_excess=np.where(g,np.maximum(0,np.abs(q-old)-radius),0),
        base_order=order(p,row_ids),final_order=order(q,row_ids))
    return q,receipt

def apply_sequence(p,meta,edges,capacities=(.1,),radius=.01):
    """Apply once per observation period / customer-out-of-fold development batch."""
    p=np.asarray(p,float);q=p.copy();receipts=[]
    folds=sorted(meta.loc[meta.fold.ge(0),'fold'].unique())
    for phase in [0,1]:
        m=meta.phase.eq(phase)&meta.fold.ge(0);sizes=np.array([(m&meta.fold.eq(f)).sum() for f in folds])
        quotas={}
        for c in capacities:
            ideal=c*sizes;v=np.floor(ideal).astype(int);n=int(np.ceil(c*sizes.sum()))-v.sum()
            v[np.lexsort((np.arange(len(folds)),-(ideal-v)))[:n]]+=1
            quotas[c]=dict(zip(folds,v))
        for fold in folds:
            ix=np.flatnonzero(m&meta.fold.eq(fold));position={r:i for i,r in enumerate(ix)}
            e=edges[edges.phase_after.eq(phase)&edges.fold.eq(fold)]
            prev=p[ix].copy();gate=np.zeros(len(ix),bool)
            for a,b,g in zip(e.row_before,e.row_after,e.seven_05):
                z=position[int(b)];prev[z]=q[int(a)];gate[z]=bool(g)
            qq,rr=score_batch(p[ix],prev,gate,meta.iloc[ix].row_id.to_numpy(),quota=[int(quotas[c][fold]) for c in capacities],radius=radius)
            q[ix]=qq
            for j,k in enumerate(ix):receipts.append(dict(matrix_row=int(k),phase=phase,fold=int(fold),sample_id=meta.iloc[k].sample_id,
                customer_id=int(meta.iloc[k].customer_id),base=float(p[k]),previous=float(prev[j]),gate=bool(gate[j]),
                temporal_candidate=float(rr['temporal_candidate'][j]),final=float(qq[j]),guard_changed=bool(rr['guard_changed'][j]),
                excess_pp=float(100*rr['unavoidable_excess'][j]),boundary_tie=bool(rr['boundary_tie'])))
    return q,receipts
