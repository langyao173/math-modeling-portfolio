"""Constrained virtual style transfer with explicit chemical defect bounds.

Chemical feasibility is a statement about the supplied measurements and chosen
thresholds. A constraint on a learned sensory surrogate is not a tasting result.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,joblib
import numpy as np,pandas as pd
from scipy.optimize import minimize,Bounds,LinearConstraint
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV,LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score,mean_absolute_error
from conditional_model import ConditionalCopula,DERIVED,TECHNICAL,EmpiricalNormalMap
from run_q3 import SEED,DEFAULT_DATA

OUT=Path(__file__).parent
DEFECTS={
 'DMS':dict(unit='ug/L',threshold=.0+19.,transform='log10',source='8.xlsx / Sensory thresholds / threshold_median'),
 'X2.3.butanedione':dict(unit='mg/L',threshold=.09,transform='identity',source='8.xlsx / Sensory thresholds / threshold_median'),
 'X4VG':dict(unit='mg/L',threshold=.2,transform='log10',source='8.xlsx / Sensory thresholds / threshold_median'),
 'X.E..2.nonenal':dict(unit='mg/L',threshold=.0001,transform='identity',source='Meilgaard 1993 DOI:10.1016/0950-3293(93)90158-3; 0.1 ug/L = 0.0001 mg/L'),
}

def as_raw(values,info):
 return np.power(10.,values) if info['transform']=='log10' else values

def enrich(frame,df,features,map_,models):
 frame=frame.copy();z=map_.transform(frame[features])
 for col,reg in models.items():frame[col]=reg.predict(z)
 raw={c:np.power(10.,frame[c].to_numpy(float)) for c in ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']}
 kcal=5.523*raw['ethanol..v.v.']+.4*(raw['GFS']+raw['unfermentables']+raw['protein.g.L.'])+.0003*raw['lactic_acid.mg.L.']
 frame['kcalperc']=np.log10(kcal)
 for tech in TECHNICAL:
  if tech not in frame:frame[tech]=df[tech].median()
 return frame

def main():
 df=pd.read_csv(DEFAULT_DATA/'1.csv');sens=pd.read_csv(DEFAULT_DATA/'4.csv').set_index('beer').loc[df.beer]
 features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL];x=df[features].to_numpy(float);styles=df.tasting_category_fine.to_numpy(str)
 control=ConditionalCopula(kind='pooled',kappa=2,nu=20).fit(x,styles)
 z=control.z_;p=z.shape[1];stale=sens.stale_hops.to_numpy(float)
 ridge=RidgeCV(alphas=[.1,1,10,100,1000]).fit(z,stale)
 # Honest internal out-of-fold performance of the sensory safety surrogate.
 oof=np.zeros(len(df));summary_oof={c:np.full(len(df),np.nan) for c in DERIVED if c!='kcalperc'}
 for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED+710).split(x,styles):
  mp=EmpiricalNormalMap().fit(x[tr]);zt=mp.transform(x[tr]);zv=mp.transform(x[va])
  r=RidgeCV(alphas=[.1,1,10,100,1000]).fit(zt,stale[tr]);oof[va]=r.predict(zv)
  for col in summary_oof:
   yy=df[col].to_numpy(float);ok=np.isfinite(yy[tr])
   reg=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=1).fit(zt[ok],yy[tr][ok])
   summary_oof[col][va]=reg.predict(zv)
 pd.DataFrame({'beer':df.beer,'observed_stale_hops_z':stale,'oof_predicted_stale_hops_z':oof}).to_csv(OUT/'stale_proxy_oof.csv',index=False)
 models={};quality=[]
 for col in summary_oof:
  yy=df[col].to_numpy(float);ok=np.isfinite(yy)
  reg=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=1).fit(z[ok],yy[ok]);models[col]=reg
  quality.append(dict(derived_column=col,oof_r2=r2_score(yy[ok],summary_oof[col][ok]),oof_mae=mean_absolute_error(yy[ok],summary_oof[col][ok]),n=int(ok.sum()),status='auxiliary predicted index; original defining formula is unavailable'))
 pd.DataFrame(quality).to_csv(OUT/'derived_index_validation.csv',index=False)
 joblib.dump(dict(model=control,features=features,stale_proxy=ridge,derived_predictors=models),OUT/'transfer_and_augmentation_models.joblib')
 target='Tripel';mu=control.mean_[target];precision=np.linalg.inv(control.cov_[target]);a=ridge.coef_
 lower=control.map_.lower_.copy();global_upper=control.map_.upper_.copy()
 clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(z,styles)
 ti=list(clf.classes_).index(target)
 srcids=np.flatnonzero(styles=='Pils/Lager');rows=[];spectra=[];defect_records=[];change_rows=[]
 for idx in srcids:
  beer=int(df.beer.iloc[idx]);z0=z[idx];x0=x[idx];upper=global_upper.copy();caps={}
  for col,info in DEFECTS.items():
   j=features.index(col);obs=x0[j];threshold=info['threshold'];raw0=as_raw(obs,info)
   # Unknown baseline: force zero exceedance; never certify an imputed baseline.
   cap=max(float(raw0),threshold) if np.isfinite(obs) else threshold
   field_cap=np.log10(cap) if info['transform']=='log10' else cap
   cap_row=control.x_[idx].copy();cap_row[j]=field_cap
   upper[j]=min(upper[j],control.map_.transform(cap_row[None])[0,j]);caps[col]=(cap,field_cap)
  for lam in [.05,.2,1.,5.]:
   h=precision+lam*np.eye(p);g=precision@mu+lam*z0
   objective=lambda u:float(.5*u@h@u-g@u)
   gradient=lambda u:h@u-g
   start=np.minimum(np.maximum(z0,lower),upper)
   ans=minimize(objective,start,jac=gradient,method='SLSQP',bounds=Bounds(lower,upper),constraints=[LinearConstraint(a,-np.inf,float(a@z0))],options={'maxiter':600,'ftol':1e-9})
   znew=ans.x;xnew=control.map_.inverse_transform(znew[None])[0]
   # Remove any last-bit interpolation overshoot in physical constraints.
   for col,(cap,field_cap) in caps.items():xnew[features.index(col)]=min(xnew[features.index(col)],field_cap)
   znew=control.map_.transform(xnew[None])[0]
   chemical_violation=0.
   for col,info in DEFECTS.items():
    j=features.index(col);raw0=as_raw(x0[j],info);rawnew=as_raw(xnew[j],info);t=info['threshold']
    old_excess=max(raw0/t-1,0) if np.isfinite(raw0) else np.nan;new_excess=max(rawnew/t-1,0)
    permitted=max(old_excess,0) if np.isfinite(old_excess) else 0.
    viol=max(new_excess-permitted,0);chemical_violation=max(chemical_violation,viol)
    defect_records.append(dict(beer=beer,lambda_preservation=lam,chemical=col,unit=info['unit'],threshold=t,source_concentration=raw0,transferred_concentration=rawnew,source_exceedance=old_excess,transferred_exceedance=new_excess,source_observed=bool(np.isfinite(x0[j])),violation=viol))
   stale0=float(ridge.predict(z0[None])[0]);stalenew=float(ridge.predict(znew[None])[0])
   source_distance=np.sqrt((z0-mu)@precision@(z0-mu)/p)
   new_distance=np.sqrt((znew-mu)@precision@(znew-mu)/p)
   targ=z[styles==target];nearest_source=float(cdist(z0[None],targ).min()/np.sqrt(p));nearest_new=float(cdist(znew[None],targ).min()/np.sqrt(p))
   row=dict(beer=beer,source_style='Pils/Lager',target_style=target,lambda_preservation=lam,optimizer_success=bool(ans.success),optimizer_message=ans.message,iterations=int(ans.nit),source_target_mahalanobis=source_distance,transferred_target_mahalanobis=new_distance,target_distance_reduction=1-new_distance/source_distance,intervention_rms=float(np.linalg.norm(znew-z0)/np.sqrt(p)),source_nearest_target=nearest_source,transferred_nearest_target=nearest_new,source_target_probability=float(clf.predict_proba(z0[None])[0,ti]),transferred_target_probability=float(clf.predict_proba(znew[None])[0,ti]),transferred_predicted_style=clf.predict(znew[None])[0],source_stale_proxy=stale0,transferred_stale_proxy=stalenew,max_chemical_exceedance_violation=chemical_violation,stale_proxy_violation=max(stalenew-stale0,0.))
   rows.append(row)
   spectrum=dict(beer=beer,lambda_preservation=lam,tasting_category_fine=target,**dict(zip(features,xnew)))
   for tech in TECHNICAL:spectrum[tech]=df.loc[idx,tech]
   spectra.append(spectrum)
   if lam==.2:
    for j,col in enumerate(features):change_rows.append(dict(beer=beer,chemical=col,source_value=x0[j],transferred_value=xnew[j],rank_space_change=znew[j]-z0[j]))
  print('Transferred beer',beer,flush=True)
 results=pd.DataFrame(rows);results.to_csv(OUT/'transfer_tradeoff.csv',index=False)
 spect=enrich(pd.DataFrame(spectra),df,features,control.map_,models)
 spect.to_csv(OUT/'transfer_all_profiles.csv',index=False)
 spect[spect.lambda_preservation==.2].to_csv(OUT/'transfer_recommended_profiles.csv',index=False)
 pd.DataFrame(defect_records).to_csv(OUT/'transfer_defect_certificate.csv',index=False)
 pd.DataFrame(change_rows).to_csv(OUT/'transfer_changes.csv',index=False)
 # Add exact caloric reconstruction and labelled auxiliary indexes to final files.
 for file in ['revised_generated_chemical_profiles.csv','revised_style_prototypes.csv']:
  if (OUT/file).exists():
   frame=pd.read_csv(OUT/file);frame=enrich(frame,df,features,control.map_,models)
   frame.to_csv(OUT/file.replace('.csv','_complete.csv'),index=False)
 meta=dict(n_source=11,n_tradeoff_solutions=len(results),main_lambda=.2,control_dimensions=p,technical_field_preserved=TECHNICAL,stale_proxy_oof_r2=r2_score(stale,oof),stale_proxy_oof_mae=mean_absolute_error(stale,oof),max_chemical_violation=float(results.max_chemical_exceedance_violation.max()),max_stale_proxy_violation=float(results.stale_proxy_violation.max()),all_solver_success=bool(results.optimizer_success.all()),stale_guarantee='nonincrease of a fitted linear surrogate only; actual sensory non-worsening requires new tasting',thresholds=DEFECTS,derived_indices='four sums predicted separately; kcalperc uses source-verified identity',method='convex quadratic target distance plus source preservation, marginal support bounds, four chemical exceedance caps, affine stale_hops surrogate constraint')
 (OUT/'transfer_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
