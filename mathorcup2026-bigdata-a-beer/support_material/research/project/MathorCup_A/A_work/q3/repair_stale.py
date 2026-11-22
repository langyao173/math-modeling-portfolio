"""Repair cross-model stale-hops violations using explicit safety margins.

Candidate selection uses the original quadratic objective among candidates that
pass both proxy checks. Its scope remains model-feasible chemical design.
"""
from pathlib import Path
import os,sys,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,pandas as pd,joblib
from scipy.optimize import minimize,Bounds,LinearConstraint
from sklearn.linear_model import LogisticRegression
from sensory_profiles import Q1DIR,with_ids
from sensory_model import predict_bundle
from transfer_profiles import OUT,DEFECTS,as_raw,enrich
from conditional_model import TECHNICAL
from run_q3 import DEFAULT_DATA,SEED

def main():
 df=pd.read_csv(DEFAULT_DATA/'1.csv');orig=df.set_index('beer');cal=pd.read_excel(DEFAULT_DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
 pack=joblib.load(OUT/'transfer_and_augmentation_models.joblib');control=pack['model'];features=pack['features'];linear=pack['stale_proxy'];models=pack['derived_predictors'];p=len(features)
 q1=joblib.load(Q1DIR/'sensory_ensemble.joblib');base=predict_bundle(q1,df);base.index=df.beer
 prior=pd.read_csv(OUT/'transfer_cross_checked_profiles.csv').set_index('beer');diag=pd.read_csv(OUT/'transfer_cross_checked_diagnostics.csv').set_index('beer')
 failed=diag.index[~diag.nontrivial_transfer];mu=control.mean_['Tripel'];precision=np.linalg.inv(control.cov_['Tripel']);lam=.2;h=precision+lam*np.eye(p);a=linear.coef_
 candidates=[];records=[]
 for beer in failed:
  src=orig.loc[beer];x0=src[features].to_numpy(float);z0=control.map_.transform(x0[None])[0];upper=control.map_.upper_.copy();lower=control.map_.lower_;g=precision@mu+lam*z0
  for col,info in DEFECTS.items():
   j=features.index(col);cap=x0[j] if np.isfinite(x0[j]) else (np.log10(info['threshold']) if info['transform']=='log10' else info['threshold'])
   row=control.map_.impute(x0[None])[0];row[j]=cap;upper[j]=min(upper[j],control.map_.transform(row[None])[0,j])
  for margin in [.025,.05,.1,.2,.4,.8]:
   obj=lambda zz:float(.5*zz@h@zz-g@zz);jac=lambda zz:h@zz-g
   ans=minimize(obj,np.clip(z0,lower,upper),jac=jac,method='SLSQP',bounds=Bounds(lower,upper),constraints=[LinearConstraint(a,-np.inf,float(a@z0-margin))],options=dict(maxiter=800,ftol=1e-9))
   xx=control.map_.inverse_transform(ans.x[None])[0]
   for col,info in DEFECTS.items():
    j=features.index(col);cap=x0[j] if np.isfinite(x0[j]) else (np.log10(info['threshold']) if info['transform']=='log10' else info['threshold']);xx[j]=min(xx[j],cap)
   row=pd.DataFrame([dict(zip(features,xx))]);row=enrich(row,df,features,control.map_,models)
   for tech in TECHNICAL:row.loc[0,tech]=src[tech]
   item=row.iloc[0].to_dict();item.update(beer=int(beer),margin=margin,alpha=1.,method='cross_model_margin_qp',tasting_category_fine='Tripel');candidates.append(item)
   zz=control.map_.transform(xx[None])[0]
   records.append(dict(beer=int(beer),margin=margin,optimizer_success=bool(ans.success),objective=float(obj(zz)),target_mahalanobis=np.sqrt((zz-mu)@precision@(zz-mu)/p),intervention_rms=np.linalg.norm(zz-z0)/np.sqrt(p),linear_delta=float(linear.predict(zz[None])[0]-linear.predict(z0[None])[0])))
 if candidates:
  cand=pd.DataFrame(candidates);pred=predict_bundle(q1,cand)
  for j,rec in enumerate(records):
   rec['ensemble_delta']=float(pred.loc[j,'stale_hops']-base.loc[rec['beer'],'stale_hops']);rec['feasible']=bool(rec['optimizer_success'] and rec['linear_delta']<=1e-8 and rec['ensemble_delta']<=1e-8)
  for beer in failed:
   indices=[j for j,r in enumerate(records) if r['beer']==beer and r['feasible']]
   if indices:
    best=min(indices,key=lambda j:records[j]['objective']);prior.loc[beer,list(cand.columns.drop('beer'))]=cand.loc[best,list(cand.columns.drop('beer'))]
  pd.DataFrame(records).to_csv(OUT/'stale_margin_search.csv',index=False);cand.to_csv(OUT/'stale_margin_candidates.csv',index=False)
 final=prior.reset_index();fp,fg=predict_bundle(q1,final,include_grades=True)
 final.to_csv(OUT/'transfer_final_profiles.csv',index=False);with_ids(final,fp).to_csv(OUT/'transfer_final_sensory_standardized.csv',index=False);with_ids(final,fg).to_csv(OUT/'transfer_final_sensory_relative_grades.csv',index=False)
 clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(control.z_,df.tasting_category_fine);ti=list(clf.classes_).index('Tripel');diagnostics=[];cert=[];sensorydelta=[]
 for i,row in final.iterrows():
  beer=int(row.beer);src=orig.loc[beer];z0=control.map_.transform(src[features].to_numpy(float)[None])[0];zn=control.map_.transform(row[features].to_numpy(float)[None])[0];d0=np.sqrt((z0-mu)@precision@(z0-mu)/p);dn=np.sqrt((zn-mu)@precision@(zn-mu)/p)
  diagnostics.append(dict(beer=beer,method=row.method,alpha=row.alpha,margin=row.get('margin',np.nan),nontrivial_transfer=bool(row.alpha>0),source_target_mahalanobis=d0,target_mahalanobis=dn,target_distance_reduction=1-dn/d0,intervention_rms=np.linalg.norm(zn-z0)/np.sqrt(p),source_stale_ensemble=float(base.loc[beer,'stale_hops']),transferred_stale_ensemble=float(fp.loc[i,'stale_hops']),ensemble_stale_delta=float(fp.loc[i,'stale_hops']-base.loc[beer,'stale_hops']),linear_stale_delta=float(linear.predict(zn[None])[0]-linear.predict(z0[None])[0]),predicted_style=clf.predict(zn[None])[0],target_probability=clf.predict_proba(zn[None])[0,ti]))
  for col,info in DEFECTS.items():
   represent=as_raw(src[col],info);after=as_raw(row[col],info);t=info['threshold'];reported=cal.loc[beer,col] if col in cal else np.nan
   if not np.isfinite(src[col]):status='missing'
   elif info['transform']=='log10' and (not np.isfinite(reported) or reported<=0):status='zero_or_nonpositive_replacement'
   else:status='positive_reported_measurement'
   before=represent if status=='positive_reported_measurement' else np.nan
   if status=='positive_reported_measurement':statement='reported positive measurement does not increase'
   elif status=='zero_or_nonpositive_replacement':statement='replacement-scale representation does not increase; generated level remains below threshold'
   else:statement='generated level is below threshold; unknown original concentration is not certified'
   cert.append(dict(beer=beer,chemical=col,unit=info['unit'],source_status=status,source_concentration=before,source_model_representative=represent,calibrated_sheet_reported_value=reported,transferred_concentration=after,threshold=t,positive_measurement_delta=after-before if np.isfinite(before) else np.nan,model_representative_delta=after-represent if np.isfinite(represent) else np.nan,transferred_excess=max(after/t-1,0) if np.isfinite(after) else np.nan,certificate=statement))
  for target in fp: sensorydelta.append(dict(beer=beer,target=target,source_prediction=base.loc[beer,target],transferred_prediction=fp.loc[i,target],delta=fp.loc[i,target]-base.loc[beer,target]))
 d=pd.DataFrame(diagnostics);d.to_csv(OUT/'transfer_final_diagnostics.csv',index=False);pd.DataFrame(cert).to_csv(OUT/'transfer_final_chemical_certificate.csv',index=False);pd.DataFrame(sensorydelta).to_csv(OUT/'transfer_final_all_sensory_changes.csv',index=False)
 meta=dict(n_sources=len(final),n_nontrivial=int(d.nontrivial_transfer.sum()),n_unmodified=int((~d.nontrivial_transfer).sum()),n_margin_repairs=int((d.method=='cross_model_margin_qp').sum()),target_distance_reduction_mean=float(d.target_distance_reduction.mean()),intervention_rms_mean=float(d.intervention_rms.mean()),ensemble_stale_max_delta=float(d.ensemble_stale_delta.max()),linear_stale_max_delta=float(d.linear_stale_delta.max()),classified_as_tripel=int((d.predicted_style=='Tripel').sum()),source_chemical_status_counts=pd.DataFrame(cert).source_status.value_counts().to_dict(),evidence_scope='constraints on reported measurements or replacement-scale representations and two fitted proxies; no independent sensory or brewing experiment')
 (OUT/'final_transfer_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
