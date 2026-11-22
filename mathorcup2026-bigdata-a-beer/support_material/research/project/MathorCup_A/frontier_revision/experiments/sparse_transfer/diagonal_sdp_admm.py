"""Bounded-time, one-CPU approximate max-trace diagonal SDP.

max 1^T d, d>=0, P-diag(d)>=0.
Standard two-block ADMM supplies candidates; a separate conservative scaling
makes the exported D primal feasible. A separately repaired W>=0 with
diag(W)>=1 gives a genuine SDP dual upper bound trace(P W). Approximate
iterations are never mislabeled an exact optimal diagonal decomposition.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve_triangular
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent


def eigmin(A):
    return float(eigh(A,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0])


def psd_projection(A):
    values,vectors=eigh((A+A.T)/2,check_finite=False)
    return (vectors*np.maximum(values,0.))@vectors.T


def primal_feasible(P,Cinv,d,eta=1e-8):
    d=np.maximum(np.asarray(d),0.)
    B=(Cinv*d)@Cinv.T
    maximum=float(eigh((B+B.T)/2,subset_by_index=[len(P)-1,len(P)-1],eigvals_only=True,check_finite=False)[0])
    factor=min(1.,(1-eta)/maximum) if maximum>0 else 1.
    safe=d*factor
    R=P-np.diag(safe)
    minimum=eigmin(R)
    if minimum<=0:
        # This conservative fallback changes D, preserving P=R+D exactly.
        for _ in range(8):
            factor*=1-1e-6
            safe=d*factor;R=P-np.diag(safe);minimum=eigmin(R)
            if minimum>0:break
    if minimum<=0:raise ArithmeticError('Could not certify positive residual matrix.')
    return safe,R,dict(scale=factor,generalized_eigen_max=maximum,R_eigen_min=minimum)


def dual_feasible(P,Y,eta=1e-8):
    W=psd_projection(Y)
    adjustment=max(eta,1+eta-float(np.diag(W).min()))
    W=W+adjustment*np.eye(len(P))
    minimum=eigmin(W)
    if minimum<0:
        W+=(-minimum+eta)*np.eye(len(P))
        minimum=eigmin(W)
    assert minimum>=0 and np.diag(W).min()>=1
    return W,dict(objective=float(np.sum(P*W.T)),W_eigen_min=minimum,
                  W_diagonal_min=float(np.diag(W).min()),diagonal_repair=adjustment)


def solve(P,seconds=120.,max_iterations=20000):
    P=(P+P.T)/2
    C=np.linalg.cholesky(P)
    Cinv=solve_triangular(C,np.eye(len(P)),lower=True,check_finite=False)
    eigen_min=eigmin(P)
    base=np.full(len(P),eigen_min*(1-1e-8))
    best_d,best_R,best_info=primal_feasible(P,Cinv,base)
    best_W,dual_info=dual_feasible(P,np.eye(len(P)))
    best_dual=dual_info['objective']
    R=P.copy();U=np.zeros_like(P);d=np.zeros(len(P));rho=1.
    records=[];start=time.perf_counter();reason='iteration_limit'
    for iteration in range(1,max_iterations+1):
        old_R=R
        d=np.maximum(0.,np.diag(P-R-U)+1/rho)
        R=psd_projection(P-np.diag(d)-U)
        primal_residual=R+np.diag(d)-P
        U=U+primal_residual
        primal_norm=float(np.linalg.norm(primal_residual,'fro'))
        dual_norm=float(rho*np.linalg.norm(np.diag(R-old_R)))
        if iteration%25==0 or iteration==1:
            candidate,candidate_R,info=primal_feasible(P,Cinv,d)
            if candidate.sum()>best_d.sum():best_d,best_R,best_info=candidate,candidate_R,info
            W,info_dual=dual_feasible(P,rho*U)
            if info_dual['objective']<best_dual:best_W,best_dual,dual_info=W,info_dual['objective'],info_dual
            gap=best_dual-float(best_d.sum())
            records.append(dict(iteration=iteration,seconds=time.perf_counter()-start,rho=rho,
                primal_residual_fro=primal_norm,dual_residual_norm=dual_norm,
                primal_trace=float(best_d.sum()),dual_bound=best_dual,absolute_gap=gap,
                relative_gap=gap/max(1.,abs(best_dual))))
            print(json.dumps(records[-1]),flush=True)
            if gap<=1e-5*max(1.,abs(best_dual)):
                reason='certified_relative_gap';break
        if time.perf_counter()-start>=seconds:
            reason='time_limit';break
        if iteration%25==0:
            old_rho=rho
            if primal_norm>10*max(dual_norm,1e-12):rho=min(rho*2,1e6)
            elif dual_norm>10*max(primal_norm,1e-12):rho=max(rho/2,1e-6)
            U*=old_rho/rho
    # One final repaired candidate and feasible dual, independent of stopping.
    candidate,candidate_R,info=primal_feasible(P,Cinv,d)
    if candidate.sum()>best_d.sum():best_d,best_R,best_info=candidate,candidate_R,info
    W,info_dual=dual_feasible(P,rho*U)
    if info_dual['objective']<best_dual:best_W,best_dual,dual_info=W,info_dual['objective'],info_dual
    L=np.linalg.cholesky(best_R).T
    audit=dict(method='ADMM_approximate_max_trace_diagonal_SDP',algorithm='Standard two-block ADMM with PSD projection',
        objective='max sum(d), d>=0, P-diag(d) PSD',dual='min trace(P W), W PSD, diag(W)>=1',
        reason=reason,iterations=iteration,seconds=time.perf_counter()-start,time_budget_seconds=seconds,
        guard=1e-8,primal_trace=float(best_d.sum()),dual_trace_upper_bound=best_dual,
        absolute_gap=best_dual-float(best_d.sum()),relative_gap=(best_dual-float(best_d.sum()))/max(1.,abs(best_dual)),
        lambda_identity_trace=float(base.sum()),trace_ratio_vs_lambda=float(best_d.sum()/base.sum()),
        diagonal_min=float(best_d.min()),diagonal_max=float(best_d.max()),
        P_eigen_min=eigen_min,R_eigen_min=eigmin(best_R),
        split_identity_max=float(np.max(abs(P-best_R-np.diag(best_d)))),
        factor_identity_max=float(np.max(abs(best_R-L.T@L))),safe_scaling=best_info,dual_feasibility=dual_info,
        certificate_note='Feasible primal/dual matrices with explicit floating-point eigenvalue checks; no exact-arithmetic SDP certificate.',
        bound_note='Larger trace does not guarantee coordinatewise or finite-time MIQCP domination; case-matched tests remain necessary.')
    assert audit['absolute_gap']>=-1e-5
    return dict(P=P,diagonal=best_d,R=best_R,factor=L,dual_matrix=best_W),audit,records


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--source',type=Path,default=HERE/'perspective/strict/beer_124/diagonal_split.npz')
    parser.add_argument('--output',type=Path,default=HERE/'perspective_sdp')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    P=np.load(args.source)['P']
    with threadpool_limits(limits=1):
        arrays,audit,history=solve(P,args.seconds)
    np.savez_compressed(args.output/'diagonal_sdp.npz',**arrays)
    (args.output/'diagonal_sdp_audit.json').write_text(json.dumps(audit,indent=2))
    pd.DataFrame(history).to_csv(args.output/'diagonal_sdp_iterations.csv',index=False)
    print('COMPLETE '+json.dumps(audit),flush=True)
