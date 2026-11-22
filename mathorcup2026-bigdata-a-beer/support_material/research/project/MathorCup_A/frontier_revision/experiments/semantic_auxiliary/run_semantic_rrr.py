"""Train-only review semantics in penalized reduced-rank regression.

 At inference ONLY attachment 1
is observed. Every semantic permutation, imputation, scale and rank fit is local
to a training partition. Attachment 7 is never an input on validation rows.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]='1'
from pathlib import Path
import sys,json,time,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'sparse_transfer'))
from control_problem import PROJECT,AW,file_sha
from sensory_model import ChemicalPreprocessor

SEED=20261023
ALPHAS=[10.,30.,100.,300.,1000.]
RANKS=[5,10,20,40,80]
TAUS=[0.,.25,1.,4.]
SPECS=[dict(mode=m,alpha=a,rank=r,tau=t) for m,a,r,t in
       itertools.product(['raw','rank'],ALPHAS,RANKS,TAUS) if t>0 or r<=40]


def standardize(A):
    A=np.asarray(A,float)
    count=np.sum(np.isfinite(A),axis=0)
    mean=np.divide(np.nansum(A,axis=0),count,out=np.zeros(A.shape[1]),where=count>0)
    filled=np.where(np.isfinite(A),A,mean)
    sd=np.std(filled,axis=0,ddof=1);sd[sd<1e-10]=1.
    return (filled-mean)/sd,mean,sd


class SemanticRRR:
    def __init__(self,spec,permuted=False,seed=SEED):
        self.spec=dict(spec);self.permuted=permuted;self.seed=seed

    def fit(self,X,Y,S):
        self.prep=ChemicalPreprocessor(self.spec['mode']).fit(X)
        Z=self.prep.transform(X);self.xmean=Z.mean(0);Z-=self.xmean
        yy,self.ymean,self.ysd=standardize(Y)
        ss,_,_=standardize(S)
        self.semantic_active=np.std(ss,axis=0)>1e-10
        ss=ss[:,self.semantic_active]
        if self.permuted:ss=ss[np.random.default_rng(self.seed).permutation(len(ss))]
        tau=self.spec['tau']
        T=yy if tau==0 else np.column_stack([yy,tau*np.sqrt(yy.shape[1]/ss.shape[1])*ss])
        U,s,Vt=np.linalg.svd(Z,full_matrices=False);a=self.spec['alpha']
        M=(s/np.sqrt(s*s+a))[:,None]*(U.T@T)
        _,_,Qt=np.linalg.svd(M,full_matrices=False)
        r=min(self.spec['rank'],len(Qt));Q=Qt[:r].T
        B=(Vt.T*(s/(s*s+a)))@(U.T@T)
        self.coef=(B@Q)@Q[:Y.shape[1]].T
        return self

    def predict(self,X):
        return ((self.prep.transform(X)-self.xmean)@self.coef)*self.ysd+self.ymean


def grid_predictions(X,Y,S,Xtest,permuted=False,seed=SEED):
    """One SVD per (mode, alpha, tau); nested rank projections reuse components."""
    yy,ym,ys=standardize(Y);ss,_,_=standardize(S)
    ss=ss[:,np.std(ss,axis=0)>1e-10]
    if permuted:ss=ss[np.random.default_rng(seed).permutation(len(ss))]
    lookup={tuple(s.values()):i for i,s in enumerate(SPECS)}
    out=np.empty((len(SPECS),len(Xtest),Y.shape[1]))
    for mode in ['raw','rank']:
        prep=ChemicalPreprocessor(mode).fit(X)
        Z=prep.transform(X);xm=Z.mean(0);Z-=xm;Ztest=prep.transform(Xtest)-xm
        U,s,Vt=np.linalg.svd(Z,full_matrices=False)
        zx=Ztest@Vt.T
        for tau in TAUS:
            T=yy if tau==0 else np.column_stack([yy,tau*np.sqrt(yy.shape[1]/ss.shape[1])*ss])
            uT=U.T@T
            for alpha in ALPHAS:
                M=(s/np.sqrt(s*s+alpha))[:,None]*uT
                _,_,Qt=np.linalg.svd(M,full_matrices=False)
                Q=Qt.T
                proj=(zx*(s/(s*s+alpha)))@uT@Q
                for rank in RANKS:
                    if tau==0 and rank>40:continue
                    r=min(rank,Q.shape[1])
                    out[lookup[(mode,alpha,rank,tau)]]=(proj[:,:r]@Q[:Y.shape[1],:r].T)*ys+ym
    return out


def core_indices(targets):
    # Dataset preserves Aroma (15) then Flavour (15); confirm exact names below.
    return np.array([i for i,t in enumerate(targets) if t.startswith(('A_','F_'))])


def tune(X,Y,S,seed):
    splits=list(KFold(3,shuffle=True,random_state=seed).split(X))
    out={kind:np.empty((len(SPECS),len(X),Y.shape[1])) for kind in ['aligned','permuted']}
    for inner,(tr,te) in enumerate(splits):
        for kind in out:
            out[kind][:,te]=grid_predictions(X[tr],Y[tr],S[tr],X[te],kind=='permuted',seed+inner*137)
    return out


def paired_ci(y,p,q,indices,seed=10041,B=2000):
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(B):
        j=rng.integers(0,len(y),len(y));yy=y[j][:,indices]
        den=np.sum((yy-yy.mean(0))**2,axis=0)
        d=np.sum((q[j][:,indices]-yy)**2-(p[j][:,indices]-yy)**2,axis=0)/np.maximum(den,1e-20)
        vals.append(float(np.mean(d)))
    return np.quantile(vals,[.025,.975]).tolist()


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    xx=pd.read_csv(PROJECT/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    yy=pd.read_csv(PROJECT/'data/A/4.csv').set_index('beer').loc[xx.beer]
    ss=pd.read_csv(PROJECT/'data/A/7.csv').set_index('beer').loc[xx.beer]
    targets=yy.columns[2:].tolist();features=xx.columns[3:].tolist()
    X=xx[features].to_numpy(float);Y=yy[targets].to_numpy(float);S=ss.iloc[:,2:].to_numpy(float)
    core=core_indices(targets)
    if len(core)!=30:
        # Official dictionary separates prefix aroma and flavor; fail rather than infer a slice.
        raise ValueError(f'Expected 30 named aroma/flavour targets, got {len(core)}: {targets}')
    protocol=dict(seed=SEED,outer_folds=5,inner_folds=3,n=250,p=X.shape[1],q=Y.shape[1],
        semantic_columns=S.shape[1],core_targets=[targets[i] for i in core],specifications=SPECS,
        modes=['no_aux','aligned_aux','permuted_aux'],selection_metrics=['all50','core30'],
        auxiliary_availability='Training labels only; never queried at inference or validation input',
        rrr_objective='min rank(B)<=r ||[Yscaled,tau*sqrt(q/s)*Sscaled]-Xscaled B||F^2 + alpha||B||F^2',
        X_sha256=file_sha(PROJECT/'data/A/1.csv'),Y_sha256=file_sha(PROJECT/'data/A/4.csv'),S_sha256=file_sha(PROJECT/'data/A/7.csv'))
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    family_names=[f'{f}_{metric}' for f in ['no_aux','aligned_aux','permuted_aux'] for metric in ['all50','core30']]
    preds={name:np.empty_like(Y) for name in family_names};selected=[];folds=np.empty(len(X),int)
    t0=time.perf_counter()
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(X)):
        cache=HERE/f'fold_{fold}_inner.npz'
        if cache.exists():
            saved=np.load(cache);pp={k:saved[k] for k in ['aligned','permuted']}
        else:
            pp=tune(X[tr],Y[tr],S[tr],SEED+fold);np.savez_compressed(cache,**pp)
        scale=np.maximum(Y[tr].std(0,ddof=1),1e-10)
        for family in ['no_aux','aligned_aux','permuted_aux']:
            kind='permuted' if family=='permuted_aux' else 'aligned'
            choices=np.array([i for i,s in enumerate(SPECS) if (s['tau']==0 if family=='no_aux' else s['tau']>0)])
            for metric,idx in [('all50',np.arange(Y.shape[1])),('core30',core)]:
                scores=np.mean(((pp[kind][:,:,idx]-Y[tr][None,:,idx])/scale[idx])**2,axis=(1,2))
                best=int(choices[np.argmin(scores[choices])]);spec=SPECS[best]
                model=SemanticRRR(spec,kind=='permuted',SEED+1000+fold).fit(X[tr],Y[tr],S[tr])
                name=f'{family}_{metric}';preds[name][te]=model.predict(X[te])
                selected.append(dict(fold=fold,family=name,spec=spec,inner_nmse=float(scores[best])))
                joblib.dump(model,HERE/f'fold_{fold}_{name}.joblib')
        folds[te]=fold
        np.savez_compressed(HERE/f'fold_{fold}_outer.npz',Y=Y[te],test_indices=te,**{k:v[te] for k,v in preds.items()})
        print(json.dumps(dict(fold=fold,elapsed_seconds=time.perf_counter()-t0,
            scores={k:float(r2_score(Y[te],p[te])) for k,p in preds.items()})),flush=True)
    baseline=np.load(AW/'q1/oof_arrays.npz')['nested_ensemble']
    assert np.array_equal(folds,pd.read_csv(AW/'q1/fold_assignments.csv').fold)
    rows=[];dim=[]
    for name,p in preds.items():
        r=r2_score(Y,p,multioutput='raw_values')
        rows.append(dict(model=name,all50_r2=float(r.mean()),core30_r2=float(r[core].mean()),
                         delta_vs_frozen_all50=float(r.mean()-r2_score(Y,baseline)),
                         ci_vs_frozen_all50=paired_ci(Y,p,baseline,np.arange(50)),
                         ci_vs_frozen_core30=paired_ci(Y,p,baseline,core)))
        dim += [dict(model=name,target=t,r2=float(r[i])) for i,t in enumerate(targets)]
    pd.DataFrame(rows).to_csv(HERE/'model_comparison.csv',index=False)
    pd.DataFrame(dim).to_csv(HERE/'dimension_metrics.csv',index=False)
    pd.DataFrame({'beer':xx.beer,'fold':folds}).to_csv(HERE/'fold_assignments.csv',index=False)
    np.savez_compressed(HERE/'oof_predictions.npz',Y=Y,beer=xx.beer,fold=folds,**preds)
    (HERE/'nested_selection.json').write_text(json.dumps(selected,indent=2))
    contrasts=[]
    for metric in ['all50','core30']:
        for ref in ['no_aux','permuted_aux']:
            p=preds[f'aligned_aux_{metric}'];q=preds[f'{ref}_{metric}']
            contrasts.append(dict(selected_for=metric,contrast=f'aligned_aux minus {ref}',
                all50_delta=float(r2_score(Y,p)-r2_score(Y,q)),all50_ci=paired_ci(Y,p,q,np.arange(50)),
                core30_delta=float(r2_score(Y[:,core],p[:,core])-r2_score(Y[:,core],q[:,core])),core30_ci=paired_ci(Y,p,q,core)))
    (HERE/'semantic_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    print(json.dumps(dict(done=True,total_seconds=time.perf_counter()-t0,models=rows)),flush=True)


if __name__=='__main__':main()
