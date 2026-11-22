"""Exact four-ceiling conditioning of a fitted interval Gaussian copula.

The returned law is P_s(X | B), never the unconditional P_s(X). B contains
specified concentration screening ceilings only; it is not a brewing or
human-tasting feasibility certificate. The four-dimensional marginal is
sampled by rejection. The other coordinates use its exact Gaussian
conditional law and are NOT independently clipped.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri

CEILINGS={
 'DMS':dict(unit='ug/L',threshold=19.,transform='log10'),
 'X2.3.butanedione':dict(unit='mg/L',threshold=.09,transform='identity'),
 'X4VG':dict(unit='mg/L',threshold=.2,transform='log10'),
 'X.E..2.nonenal':dict(unit='mg/L',threshold=.0001,transform='identity'),
}

def feasibility(x,features,ceilings=CEILINGS):
    x=np.asarray(x,float);checks=[]
    for field,info in ceilings.items():
        j=features.index(field)
        t=np.log10(info['threshold']) if info['transform']=='log10' else info['threshold']
        checks.append(np.isfinite(x[:,j]) & (x[:,j] <= t+1e-12))
    return np.all(checks,axis=0),np.column_stack(checks)


def wilson(successes,trials,z=1.959963984540054):
    if trials==0:return np.nan,np.nan
    phat=successes/trials;den=1+z*z/trials
    center=(phat+z*z/(2*trials))/den
    radius=z*np.sqrt(phat*(1-phat)/trials+z*z/(4*trials*trials))/den
    return max(0.,center-radius),min(1.,center+radius)

@dataclass
class SamplingRecord:
    requested:int
    accepted:int
    proposed:int
    acceptance_estimate:float
    acceptance_ci_low:float
    acceptance_ci_high:float
    complete:bool
    min_conditional_eigenvalue:float

class CeilingConditionedIntervalGaussian:
    def __init__(self,model,features,ceilings=CEILINGS,styles=None):
        if model.kind!='interval_gaussian':
            raise ValueError('Only an interval Gaussian is supported; a kernel mixture needs component reweighting.')
        self.model=model;self.features=list(features);self.ceilings=ceilings
        indices=[];upper=[];self.bounds=[]
        for field,info in ceilings.items():
            j=self.features.index(field);values,lo,hi=model.intervals_[j]
            t=np.log10(info['threshold']) if info['transform']=='log10' else info['threshold']
            allowed=np.flatnonzero(values<=t+1e-12)
            prob=0. if not len(allowed) else float(hi[allowed[-1]])
            self.bounds.append(dict(field=field,field_ceiling=float(t),marginal_cdf=prob,latent_upper=float(ndtri(prob))))
            if prob==0:raise ValueError('No training marginal support passes the ceiling for '+field)
            if prob<1:indices.append(j);upper.append(ndtri(prob))
        self.C=np.array(indices,int);self.upper=np.array(upper,float)
        self.R=np.setdiff1d(np.arange(len(features)),self.C)
        self.cache={}
        for style in (model.styles_ if styles is None else styles):
            cov=model.cov_[style];mu=model.mean_[style]
            if len(self.C):
                cc=cov[np.ix_(self.C,self.C)]
                rc=cov[np.ix_(self.R,self.C)]
                gain=np.linalg.solve(cc,rc.T).T
                schur=cov[np.ix_(self.R,self.R)]-gain@rc.T
                schur=(schur+schur.T)/2
                eig=float(np.linalg.eigvalsh(schur)[0])
                # No additional numerical repair: the fitted covariance must
                # itself imply a positive definite conditional covariance.
                if eig<=0:raise ArithmeticError('Nonpositive conditional covariance')
                self.cache[style]=(np.linalg.cholesky(cc),gain,np.linalg.cholesky(schur),eig)
            else:self.cache[style]=(None,None,None,float(np.linalg.eigvalsh(cov)[0]))

    def estimate_acceptance(self,style,rng,n=100000):
        if not len(self.C):return dict(successes=n,trials=n,estimate=1.,ci_low=1.,ci_high=1.)
        chol=self.cache[style][0];mu=self.model.mean_[style][self.C]
        accepted=0
        for start in range(0,n,50000):
            a=mu+rng.standard_normal((min(50000,n-start),len(self.C)))@chol.T
            accepted+=int(np.all(a<=self.upper,axis=1).sum())
        low,high=wilson(accepted,n)
        return dict(successes=accepted,trials=n,estimate=accepted/n,ci_low=low,ci_high=high)

    def sample(self,style,n,rng,max_proposals=5000000,block_size=20000,return_rank=True):
        n=int(n);p=len(self.features);mu=self.model.mean_[style]
        if not len(self.C):
            x,z=self.model.sample(style,n,rng)
            return x,z,SamplingRecord(n,n,n,1.,1.,1.,True,self.cache[style][3])
        cc,gain,schur,eig=self.cache[style];accepted=[];seen=0;kept=0
        while kept<n and seen<max_proposals:
            block=min(block_size,max_proposals-seen)
            xc=mu[self.C]+rng.standard_normal((block,len(self.C)))@cc.T
            ok=np.all(xc<=self.upper,axis=1);got=xc[ok]
            accepted.append(got);kept+=len(got);seen+=block
        # All proposals in the final full block are counted in the acceptance
        # estimate. A separately fixed-budget estimate is used for inference.
        zc=np.vstack(accepted)[:n] if accepted else np.empty((0,len(self.C)))
        z=np.empty((len(zc),p));z[:,self.C]=zc
        if len(self.R):
            z[:,self.R]=mu[self.R]+(zc-mu[self.C])@gain.T+rng.standard_normal((len(zc),len(self.R)))@schur.T
        x=self.model.inverse_latent(z)
        passed,_=feasibility(x,self.features,self.ceilings)
        if not passed.all():raise ArithmeticError('Decoded concentration violates specified ceiling')
        low,high=wilson(kept,seen)
        return x,(self.model.map_.transform(x) if return_rank else z),SamplingRecord(n,len(x),seen,kept/seen if seen else np.nan,np.nan,np.nan,len(x)==n,eig)
