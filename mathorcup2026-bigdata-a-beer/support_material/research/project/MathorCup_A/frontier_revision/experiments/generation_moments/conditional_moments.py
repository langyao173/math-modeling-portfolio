"""Regularized conditional-moment refinement of the interval generator.


Intervals represent empirical tied values (including author-replaced zeros),
not instrument detection limits. Gaussian copula latent-interval inference is
established methodology; see Zhao and Udell (KDD 2020; JSS 2024).

This implementation uses finite Gibbs samples and covariance shrinkage. It is
a controlled refinement experiment, not exact maximum likelihood or a claim
of monotone EM convergence. Monte Carlo draws are never counted as new beers.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]='1'
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri
from scipy.stats import truncnorm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'sparse_transfer'))
from control_problem import AW
from conditional_model import IntervalCopula


def gibbs_sweep(z, means, precision_by_style, labels, lower, upper, rng):
    """One exact conditional Gaussian sweep, vectorized across observations."""
    ps=precision_by_style[labels]
    residual=np.einsum('nij,nj->ni',ps,z-means)
    diag=np.diagonal(ps,axis1=1,axis2=2)
    for j in rng.permutation(z.shape[1]):
        sd=1/np.sqrt(diag[:,j])
        mu=z[:,j]-residual[:,j]/diag[:,j]
        a=(lower[:,j]-mu)/sd;b=(upper[:,j]-mu)/sd
        new=truncnorm.rvs(a,b,loc=mu,scale=sd,random_state=rng)
        delta=new-z[:,j]
        z[:,j]=new
        residual+=delta[:,None]*ps[:,:,j]
    return z


class ConditionalMomentCopula(IntervalCopula):
    def __init__(self,kappa=2.,nu=20.,iterations=8,burn_sweeps=8,
                 retained_sweeps=8,missing_intervals=False,seed=20261024):
        super().__init__(kind='interval_gaussian',kappa=kappa,nu=nu)
        self.iterations=iterations;self.burn_sweeps=burn_sweeps
        self.retained_sweeps=retained_sweeps
        self.missing_intervals=missing_intervals;self.seed=seed

    def fit(self,x,style):
        super().fit(x,style)
        x=np.asarray(x,float);n,p=x.shape
        lower=ndtri(self.row_low_);upper=ndtri(self.row_high_)
        # Missing-aware ablation learns each marginal from observed entries;
        # unavailable coordinates constrain no latent value.
        if self.missing_intervals:
            self.intervals_=[]
            for j in range(p):
                obs=np.isfinite(x[:,j]);v,c=np.unique(x[obs,j],return_counts=True)
                hi=np.cumsum(c)/sum(c);lo=np.r_[0.,hi[:-1]]
                ix=np.searchsorted(v,x[obs,j])
                lower[obs,j]=ndtri(lo[ix]);upper[obs,j]=ndtri(hi[ix])
                lower[~obs,j]=-np.inf;upper[~obs,j]=np.inf
                self.intervals_.append((v,lo,hi))
        z=self.latent_.copy()
        # Map midpoint probabilities into the new boxes, including missingness.
        from scipy.special import ndtr
        mid=(ndtr(lower)+ndtr(upper))/2
        z=ndtri(np.clip(mid,1e-10,1-1e-10))
        labels=np.searchsorted(self.styles_,self.style_)
        masks=[np.flatnonzero(labels==k) for k in range(len(self.styles_))]
        alpha=self.pooled_shrinkage_
        rng=np.random.default_rng(self.seed);eye=np.eye(p)
        self.refinement_trace_=[]
        for it in range(self.iterations):
            old_pool=self.pooled_cov_.copy()
            old_means=np.stack([self.mean_[s] for s in self.styles_])
            precision=np.stack([np.linalg.inv(self.cov_[s]) for s in self.styles_])
            means=old_means[labels]
            for _ in range(self.burn_sweeps):
                z=gibbs_sweep(z,means,precision,labels,lower,upper,rng)
            mean_z=np.zeros_like(z)
            second=[np.zeros((p,p)) for _ in masks]
            for _ in range(self.retained_sweeps):
                z=gibbs_sweep(z,means,precision,labels,lower,upper,rng)
                mean_z+=z/self.retained_sweeps
                for k,idx in enumerate(masks):
                    second[k]+=z[idx].T@z[idx]/self.retained_sweeps
            grand=mean_z.mean(0)
            centers=[mean_z[idx].mean(0) for idx in masks]
            scatter=[q-len(idx)*np.outer(m,m) for q,idx,m in zip(second,masks,centers)]
            pooled=sum(scatter)/max(n-len(masks),1)
            pooled=(pooled+pooled.T)/2
            self.pooled_cov_=(1-alpha)*pooled+alpha*np.trace(pooled)/p*eye+1e-5*eye
            for k,(s,idx,m) in enumerate(zip(self.styles_,masks,centers)):
                ns=len(idx)
                self.mean_[s]=(ns*m+self.kappa*grand)/(ns+self.kappa)
                local=scatter[k]/max(ns-1,1) if ns>=2 else self.pooled_cov_
                w=(ns-1)/(ns-1+self.nu)
                cov=w*local+(1-w)*self.pooled_cov_
                cov=(cov+cov.T)/2+1e-5*eye
                self.cov_[s]=cov;self.chol_[s]=np.linalg.cholesky(cov)
            self.refinement_trace_.append(dict(iteration=it+1,
                relative_pooled_change=float(np.linalg.norm(self.pooled_cov_-old_pool)/np.linalg.norm(old_pool)),
                mean_change_rms=float(np.sqrt(np.mean((np.stack([self.mean_[s] for s in self.styles_])-old_means)**2))),
                maximum_interval_violation=float(max(0,np.max(lower-z),np.max(z-upper))),
                minimum_style_covariance_eigenvalue=float(min(np.linalg.eigvalsh(self.cov_[s])[0] for s in self.styles_))))
            self.latent_=mean_z
        self.center_={s:m for s,m in zip(self.styles_,centers)}
        return self
