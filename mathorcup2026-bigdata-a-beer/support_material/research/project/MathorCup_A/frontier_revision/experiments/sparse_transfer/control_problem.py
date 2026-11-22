"""Fixed-model sparse chemical intervention problem and independent row decoding.

 The original fixed predictors
and chemical bounds are reused so optimization changes can be compared fairly.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import sys, hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import milp, Bounds, LinearConstraint, minimize
from scipy.sparse import csr_matrix


def find_project():
    if os.environ.get('MATHORCUP_PROJECT_DIR'):
        return Path(os.environ['MATHORCUP_PROJECT_DIR']).resolve()
    for parent in Path(__file__).resolve().parents:
        for p in (parent, parent/'project/MathorCup_A'):
            if (p/'data/A/1.csv').is_file() and (p/'A_work/q1/sensory_model.py').is_file():
                return p
    raise FileNotFoundError('Set MATHORCUP_PROJECT_DIR to the restored project root.')


PROJECT=find_project()
AW=PROJECT/'A_work'
sys.path[:0]=[str(AW/'q1'),str(AW/'q3')]
from sensory_model import predict_bundle
from conditional_model import DERIVED, TECHNICAL
from transfer_profiles import DEFECTS


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


class FixedProblem:
    def __init__(self,policy='strict',style_epsilon=1e-5):
        if policy not in ('strict','exceedance'):raise ValueError(policy)
        self.policy=policy
        self.style_epsilon=float(style_epsilon)
        self.df=pd.read_csv(PROJECT/'data/A/1.csv')
        self.pack=joblib.load(AW/'q3/transfer_and_augmentation_models.joblib')
        self.control=self.pack['model']; self.features=list(self.pack['features'])
        self.bundle=joblib.load(AW/'q1/sensory_ensemble.joblib')
        self.clf=joblib.load(AW/'competition_revision/style_feasibility/frozen_style_classifier.joblib')
        self.x=self.df[self.features].to_numpy(float)
        self.z=self.control.map_.transform(self.x); self.p=self.z.shape[1]
        self.mu=self.control.mean_['Tripel']
        self.P=np.linalg.inv(self.control.cov_['Tripel'])
        self.H=self.P+.2*np.eye(self.p)
        self.a=np.asarray(self.pack['stale_proxy'].coef_)
        ti=list(self.clf.classes_).index('Tripel')
        oi=np.flatnonzero(np.arange(len(self.clf.classes_))!=ti)
        self.w=self.clf.coef_[ti]-self.clf.coef_[oi]
        self.b=self.clf.intercept_[ti]-self.clf.intercept_[oi]
        self.target_index=ti
        self.sensory0=predict_bundle(self.bundle,self.df)
        self.contexts={}
        for i in np.flatnonzero(self.df.tasting_category_fine.eq('Pils/Lager')):
            row=self.df.iloc[i]; beer=int(row.beer); z0=self.z[i]; x0=self.x[i]
            lower=self.control.map_.lower_.copy(); upper=self.control.map_.upper_.copy()
            caps={}; mandatory=[]
            for col,info in DEFECTS.items():
                j=self.features.index(col)
                threshold=np.log10(info['threshold']) if info['transform']=='log10' else info['threshold']
                cap=(x0[j] if policy=='strict' else max(x0[j],threshold)) if np.isfinite(x0[j]) else threshold
                t=self.control.map_.impute(x0[None])[0]; t[j]=cap
                upper[j]=min(upper[j],self.control.map_.transform(t[None])[0,j])
                caps[j]=float(cap)
                if not np.isfinite(x0[j]): mandatory.append(j)
            self.contexts[beer]=dict(index=int(i),src=row,z0=z0,x0=x0,lo=lower-z0,hi=upper-z0,caps=caps,
                mandatory=mandatory,base_margin=self.w@z0+self.b,
                gradient=self.P@(z0-self.mu),F0=float(.5*(z0-self.mu)@self.P@(z0-self.mu)),
                d0=float(np.sqrt((z0-self.mu)@self.P@(z0-self.mu)/self.p)),
                stale0=float(self.sensory0.iloc[i].stale_hops),
                aux0={c:float(m.predict(z0[None])[0]) for c,m in self.pack['derived_predictors'].items()})

    def linear_constraints(self,beer,margin=0.,style_epsilon=None):
        c=self.contexts[beer]
        if style_epsilon is None:style_epsilon=self.style_epsilon
        return np.vstack([self.a,-self.w]),np.r_[-margin,c['base_margin']-style_epsilon]

    def objective(self,beer,d):
        c=self.contexts[beer]
        return float(c['F0']+c['gradient']@d+.5*d@self.H@d)

    def cardinality_milp(self,beer,margin=0.,budget=None,cost=None,excluded=(),time_limit=60.):
        c=self.contexts[beer]; p=self.p; A,rhs=self.linear_constraints(beer,margin)
        blocks=[np.column_stack([A,np.zeros((len(A),p))]),
                np.column_stack([np.eye(p),-np.diag(c['hi'])]),
                np.column_stack([-np.eye(p),np.diag(c['lo'])])]
        rights=[rhs,np.zeros(p),np.zeros(p)]
        if budget is not None:
            blocks.append(np.r_[np.zeros(p),np.ones(p)][None]); rights.append(np.array([budget]))
        for supp in excluded:
            # Enumeration only. Bounds from this restricted problem are NOT
            # used as global lower bounds on the original model.
            r=np.zeros(2*p);r[p+np.asarray(supp,int)]=1
            blocks.append(r[None]);rights.append(np.array([len(supp)-1]))
        binary_low=np.zeros(p);binary_low[c['mandatory']]=1
        obj=np.r_[np.zeros(p),np.ones(p) if cost is None else np.asarray(cost)]
        return milp(obj,integrality=np.r_[np.zeros(p),np.ones(p)],
            bounds=Bounds(np.r_[np.minimum(c['lo'],0),binary_low],np.r_[np.maximum(c['hi'],0),np.ones(p)]),
            constraints=LinearConstraint(csr_matrix(np.vstack(blocks)),-np.inf,np.concatenate(rights)),
            options=dict(time_limit=float(time_limit),mip_rel_gap=0.,presolve=True))

    def polish(self,beer,support,margin=0.,initial=None,rho=None):
        c=self.contexts[beer]; s=np.asarray(sorted(set(support)),int)
        A,rhs=self.linear_constraints(beer,margin)
        H=self.H[np.ix_(s,s)];g=c['gradient'][s]
        lower,upper=c['lo'][s],c['hi'][s]
        x=np.clip(np.zeros(len(s)) if initial is None else np.asarray(initial)[s],lower,upper)
        constraints=[LinearConstraint(A[:,s],-np.inf,rhs)]
        if rho is not None:
            PP=self.P[np.ix_(s,s)];constant=2*c['F0']*(1-(1-rho)**2)
            constraints.append(dict(type='ineq',fun=lambda d:float(-constant-2*g@d-d@PP@d),
                                    jac=lambda d:-2*g-2*PP@d))
        q=minimize(lambda d:float(.5*d@H@d+g@d),x,jac=lambda d:H@d+g,
            bounds=Bounds(lower,upper),constraints=constraints,
            method='SLSQP',options=dict(maxiter=600,ftol=1e-10))
        d=np.zeros(self.p);d[s]=q.x
        violation=max(np.max(c['lo']-d),np.max(d-c['hi']),np.max(A@d-rhs))
        return q,d,float(violation)

    def decode(self,beer,d,support):
        c=self.contexts[beer];s=np.asarray(sorted(set(support)),int)
        vals=c['x0'].copy();inv=self.control.map_.inverse_transform((c['z0']+d)[None])[0]
        vals[s]=inv[s]
        for j,cap in c['caps'].items():
            if j in s:vals[j]=min(vals[j],cap)
        zz=self.control.map_.transform(vals[None])[0]
        row=c['src'].to_dict();row.update(dict(zip(self.features,vals)))
        for col,m in self.pack['derived_predictors'].items():
            new=float(m.predict(zz[None])[0]);old=float(c['src'][col])
            value=old+new-c['aux0'][col] if np.isfinite(old) else new
            row[col]=float(np.clip(value,np.nanmin(self.df[col]),np.nanmax(self.df[col])))
        raw={col:10.**row[col] for col in ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']}
        row['kcalperc']=float(np.log10(5.523*raw['ethanol..v.v.']+.4*(raw['GFS']+raw['unfermentables']+raw['protein.g.L.'])+.0003*raw['lactic_acid.mg.L.']))
        row.update(source_tasting_category_fine=str(c['src'].tasting_category_fine),requested_target_style='Tripel',profile_origin='sparse_optimization')
        return row,zz

    def check_rows(self,rows,rho=0.,margin=0.,atol=1e-8):
        frame=pd.DataFrame(rows)
        pred=predict_bundle(self.bundle,frame)
        zz=self.control.map_.transform(frame[self.features].to_numpy(float))
        labels=self.clf.predict(zz);probs=self.clf.predict_proba(zz)[:,self.target_index]
        out=[]
        for i,row in frame.iterrows():
            beer=int(row.beer);c=self.contexts[beer];d=zz[i]-c['z0']
            values=row[self.features].to_numpy(float)
            same=(values==c['x0'])|(np.isnan(values)&np.isnan(c['x0']))
            caps=max((values[j]-cap for j,cap in c['caps'].items() if np.isfinite(values[j])),default=0.)
            unresolved=sum(not np.isfinite(values[j]) for j in c['caps'])
            lin=float(self.a@d);nonlin=float(pred.iloc[i].stale_hops-c['stale0'])
            distance=float(np.sqrt((zz[i]-self.mu)@self.P@(zz[i]-self.mu)/self.p))
            reduction=1-distance/c['d0'];sm=float(np.min(c['base_margin']+self.w@d))
            box=float(max(np.max(c['lo']-d),np.max(d-c['hi'])))
            required_rho=float(row.get('required_rho',rho));required_margin=float(row.get('required_proxy_margin',margin))
            out.append(dict(beer=beer,raw_fields_changed=int((~same).sum()),rank_coordinates_changed=int((abs(d)>1e-8).sum()),
                target_distance_reduction=reduction,target_distance=distance,source_target_distance=c['d0'],
                quadratic_objective=self.objective(beer,d),intervention_rms=float(np.linalg.norm(d)/np.sqrt(self.p)),
                chemical_violation=float(caps),unresolved_missing_defects=int(unresolved),linear_stale_delta=lin,
                ensemble_stale_delta=nonlin,min_style_margin=sm,box_violation=box,predicted_style=str(labels[i]),target_probability=float(probs[i]),
                required_rho=required_rho,required_proxy_margin=required_margin,required_style_epsilon=self.style_epsilon,
                accepted=bool(caps<=atol and unresolved==0 and lin<=-required_margin+atol and nonlin<=atol and
                              sm>=self.style_epsilon-atol and box<=atol and labels[i]=='Tripel' and
                              reduction>=required_rho-atol and reduction>atol)))
        return pd.DataFrame(out),pred
