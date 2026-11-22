"""Compare unconstrained, coordinate-clipped and jointly constrained transfer.

The outputs are computational candidates; proxy agreement is not human tasting.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import joblib,json
import numpy as np,pandas as pd
from scipy.optimize import minimize,Bounds,LinearConstraint
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from transfer_profiles import OUT,DEFECTS,as_raw,enrich
from run_q3 import DEFAULT_DATA,SEED
from conditional_model import TECHNICAL

def main():
 pack=joblib.load(OUT/'transfer_and_augmentation_models.joblib');control=pack['model'];features=pack['features'];proxy=pack['stale_proxy'];models=pack['derived_predictors']
 df=pd.read_csv(DEFAULT_DATA/'1.csv');x=df[features].to_numpy(float);styles=df.tasting_category_fine.to_numpy(str);z=control.z_;p=z.shape[1]
 target='Tripel';lam=.2;mu=control.mean_[target];prec=np.linalg.inv(control.cov_[target]);h=prec+lam*np.eye(p)
 clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(z,styles);ti=list(clf.classes_).index(target)
 lower=control.map_.lower_;global_upper=control.map_.upper_
 joint=pd.read_csv(OUT/'transfer_recommended_profiles.csv').set_index('beer')
 spectra=[];metrics=[];certificates=[];sensitivities=[]
 for idx in np.flatnonzero(styles=='Pils/Lager'):
  beer=int(df.beer.iloc[idx]);z0=z[idx];x0=x[idx];g=prec@mu+lam*z0
  obj=lambda u:float(.5*u@h@u-g@u);jac=lambda u:h@u-g
  ans=minimize(obj,z0,jac=jac,method='L-BFGS-B',bounds=list(zip(lower,global_upper)),options=dict(maxiter=1000,ftol=1e-14,gtol=1e-8))
  uncon=control.map_.inverse_transform(ans.x[None])[0];clipped=uncon.copy();strictupper=global_upper.copy()
  for col,info in DEFECTS.items():
   j=features.index(col);raw0=as_raw(x0[j],info);t=info['threshold'];cap=max(raw0,t) if np.isfinite(raw0) else t
   fieldcap=np.log10(cap) if info['transform']=='log10' else cap
   clipped[j]=min(clipped[j],fieldcap)
   strictcap=x0[j] if np.isfinite(x0[j]) else (np.log10(t) if info['transform']=='log10' else t)
   rr=control.x_[idx].copy();rr[j]=strictcap;strictupper[j]=min(strictupper[j],control.map_.transform(rr[None])[0,j])
  strict=minimize(obj,np.clip(z0,lower,strictupper),jac=jac,method='SLSQP',bounds=Bounds(lower,strictupper),constraints=[LinearConstraint(proxy.coef_,-np.inf,float(proxy.coef_@z0))],options=dict(maxiter=600,ftol=1e-9))
  strictx=control.map_.inverse_transform(strict.x[None])[0]
  candidates={'unconstrained':uncon,'coordinate_clipping':clipped,'joint_qp':joint.loc[beer,features].to_numpy(float),'strict_concentration_qp':strictx}
  for mode,new in candidates.items():
   zn=control.map_.transform(new[None])[0];stale0=float(proxy.predict(z0[None])[0]);stalenew=float(proxy.predict(zn[None])[0]);viol=[];concincrease=[]
   for col,info in DEFECTS.items():
    j=features.index(col);raw0=as_raw(x0[j],info);rawnew=as_raw(new[j],info);t=info['threshold']
    oldex=max(raw0/t-1,0) if np.isfinite(raw0) else np.nan;newex=max(rawnew/t-1,0)
    v=max(newex-(oldex if np.isfinite(oldex) else 0),0);viol.append(v)
    concincrease.append(bool(np.isfinite(raw0) and rawnew>raw0+1e-8))
    certificates.append(dict(beer=beer,method=mode,chemical=col,source_concentration=raw0,transferred_concentration=rawnew,threshold=t,unit=info['unit'],source_excess=oldex,transferred_excess=newex,excess_violation=v,concentration_increase=concincrease[-1],source_observed=bool(np.isfinite(raw0))))
    alt={'DMS':[17.25,19.],'X2.3.butanedione':[.09,.29925],'X4VG':[.2,.2063333333],'X.E..2.nonenal':[.00003,.0001,.00025]}[col]
    for tt in alt:
     oe=max(raw0/tt-1,0) if np.isfinite(raw0) else np.nan;ne=max(rawnew/tt-1,0)
     sensitivities.append(dict(beer=beer,method=mode,chemical=col,threshold=tt,source_excess=oe,transferred_excess=ne,excess_violation=max(ne-(oe if np.isfinite(oe) else 0),0)))
   distance0=np.sqrt((z0-mu)@prec@(z0-mu)/p);distancen=np.sqrt((zn-mu)@prec@(zn-mu)/p)
   metrics.append(dict(beer=beer,method=mode,target_distance_reduction=1-distancen/distance0,intervention_rms=np.linalg.norm(zn-z0)/np.sqrt(p),source_target_mahalanobis=distance0,transferred_target_mahalanobis=distancen,target_probability=clf.predict_proba(zn[None])[0,ti],predicted_style=clf.predict(zn[None])[0],source_stale_proxy=stale0,transferred_stale_proxy=stalenew,stale_proxy_violation=max(stalenew-stale0,0),max_chemical_violation=max(viol),n_concentration_increases=sum(concincrease)))
   row=dict(beer=beer,method=mode,tasting_category_fine=target,**dict(zip(features,new)))
   for tech in TECHNICAL:row[tech]=df.loc[idx,tech]
   spectra.append(row)
 pd.DataFrame(metrics).to_csv(OUT/'transfer_comparison_metrics.csv',index=False)
 pd.DataFrame(certificates).to_csv(OUT/'transfer_comparison_certificate.csv',index=False)
 pd.DataFrame(sensitivities).to_csv(OUT/'transfer_threshold_sensitivity.csv',index=False)
 enriched=enrich(pd.DataFrame(spectra),df,features,control.map_,models)
 enriched.to_csv(OUT/'transfer_comparison_profiles.csv',index=False)
 # Strict policy is robust to every positive threshold on observed chemicals.
 enriched[enriched.method=='strict_concentration_qp'].to_csv(OUT/'transfer_strict_profiles.csv',index=False)
 print(pd.DataFrame(metrics).groupby('method').agg(target_distance_reduction=('target_distance_reduction','mean'),intervention=('intervention_rms','mean'),chemical_violation_count=('max_chemical_violation',lambda x:int(sum(x>1e-8))),stale_violation_count=('stale_proxy_violation',lambda x:int(sum(x>1e-8))),tripel=('predicted_style',lambda x:int(sum(x=='Tripel')))).to_string())

if __name__=='__main__':main()
