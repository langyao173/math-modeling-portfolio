"""Join Q1 sensory predictions to Q3 generation and verify transfer candidates.

The Q1 ensemble is a second model architecture trained on the same dataset;
cross-model agreement is not an independent tasting experiment.
"""
from pathlib import Path
import os,sys,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
Q1DIR=Path(os.environ.get('MATHORCUP_Q1_DIR',str(Path(__file__).parent.parent/'q1')))
sys.path.insert(0,str(Q1DIR))
from sensory_model import predict_bundle
from transfer_profiles import OUT,DEFECTS,as_raw,enrich
from conditional_model import TECHNICAL
from run_q3 import DEFAULT_DATA,SEED

def with_ids(frame,pred):
 keys=[c for c in ['synthetic_id','beer','method','tasting_category_fine','alpha'] if c in frame]
 return pd.concat([frame[keys].reset_index(drop=True),pred.reset_index(drop=True)],axis=1)

def main():
 q1bundle=joblib.load(Q1DIR/'sensory_ensemble.joblib')
 df=pd.read_csv(DEFAULT_DATA/'1.csv');base=predict_bundle(q1bundle,df);base.index=df.beer
 gen=pd.read_csv(OUT/'revised_generated_chemical_profiles_complete.csv');p,g=predict_bundle(q1bundle,gen,include_grades=True)
 with_ids(gen,p).to_csv(OUT/'generated_sensory_standardized.csv',index=False)
 with_ids(gen,g).to_csv(OUT/'generated_sensory_relative_grades.csv',index=False)
 rep=pd.read_csv(OUT/'requested_style_representatives.csv');rp,rg=predict_bundle(q1bundle,rep,include_grades=True)
 with_ids(rep,rp).to_csv(OUT/'requested_style_sensory_standardized.csv',index=False)
 with_ids(rep,rg).to_csv(OUT/'requested_style_sensory_relative_grades.csv',index=False)
 summaries=[]
 for s in gen.tasting_category_fine.unique():
  mask=gen.tasting_category_fine==s
  for target in p.columns:
   v=p.loc[mask,target];summaries.append(dict(style=s,target=target,predicted_q10=v.quantile(.1),predicted_median=v.median(),predicted_q90=v.quantile(.9),scale='standardized panel response; not raw 0-6'))
 pd.DataFrame(summaries).to_csv(OUT/'generated_sensory_fingerprint_summary.csv',index=False)
 comparison=pd.read_csv(OUT/'transfer_comparison_profiles.csv');cp=predict_bundle(q1bundle,comparison);with_ids(comparison,cp).to_csv(OUT/'transfer_comparison_sensory.csv',index=False)
 deltas=[]
 for i,row in comparison.iterrows():
  for target in cp:
   b=base.loc[int(row.beer),target];v=cp.loc[i,target];deltas.append(dict(beer=int(row.beer),method=row.method,target=target,source_prediction=b,transfer_prediction=v,delta=v-b))
 delta=pd.DataFrame(deltas);delta.to_csv(OUT/'transfer_cross_model_deltas.csv',index=False)
 stalecheck=delta[delta.target=='stale_hops'];print(stalecheck.groupby('method').delta.agg(['min','mean','max']).to_string(),flush=True)
 # Search a fixed finite segment for this diagnostic repair, between the
 # original and strict QP candidate. This is not a preregistered validation.
 pack=joblib.load(OUT/'transfer_and_augmentation_models.joblib');control=pack['model'];features=pack['features'];dermodels=pack['derived_predictors'];linear=pack['stale_proxy'];pdim=len(features)
 original=df.set_index('beer');strict=comparison[comparison.method=='strict_concentration_qp'].set_index('beer');target='Tripel';mu=control.mean_[target];prec=np.linalg.inv(control.cov_[target])
 ztrain=control.z_;clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(ztrain,df.tasting_category_fine);tripel_index=list(clf.classes_).index('Tripel')
 candidates=[]
 for beer,row in strict.iterrows():
  src=original.loc[beer];x0=src[features].to_numpy(float);z0=control.map_.transform(x0[None])[0];zt=control.map_.transform(row[features].to_numpy(float)[None])[0]
  # alpha=0 is the unmodified measured source, not silently imputed chemistry.
  zero=dict(beer=int(beer),alpha=0.,method='cross_model_checked',tasting_category_fine=src.tasting_category_fine,**{c:src[c] for c in df.columns[3:]});candidates.append(zero)
  for alpha in np.arange(1,81)/80.:
   zz=(1-alpha)*z0+alpha*zt;xx=control.map_.inverse_transform(zz[None])[0]
   for col,info in DEFECTS.items():
    j=features.index(col)
    if np.isfinite(x0[j]):xx[j]=min(xx[j],x0[j])
    else:
     cap=info['threshold'];xx[j]=min(xx[j],np.log10(cap) if info['transform']=='log10' else cap)
   new=pd.DataFrame([dict(zip(features,xx))]);new=enrich(new,df,features,control.map_,dermodels)
   # Preserve the measured baseline of an auxiliary derived index at alpha=0.
   for col,reg in dermodels.items():
    if np.isfinite(src[col]):new.loc[0,col]+=(1-alpha)*(src[col]-reg.predict(z0[None])[0])
   for tech in TECHNICAL:new.loc[0,tech]=src[tech]
   item=new.iloc[0].to_dict();item.update(beer=int(beer),alpha=float(alpha),method='cross_model_checked',tasting_category_fine=target);candidates.append(item)
 cand=pd.DataFrame(candidates);pred=predict_bundle(q1bundle,cand)
 selected=[];diagnostics=[];allchecks=[]
 for beer in strict.index:
  ix=np.flatnonzero(cand.beer.to_numpy()==beer);src=original.loc[beer];z0=control.map_.transform(src[features].to_numpy(float)[None])[0];source_linear=float(linear.predict(z0[None])[0]);source_full=float(base.loc[beer,'stale_hops'])
  feasible=[]
  for j in ix:
   znew=control.map_.transform(cand.loc[[j],features])[0]
   qlinear=float(linear.predict(znew[None])[0]);qfull=float(pred.loc[j,'stale_hops']);a=float(cand.loc[j,'alpha']);is_feasible=(qlinear<=source_linear+1e-8 and qfull<=source_full+1e-8)
   allchecks.append(dict(beer=int(beer),alpha=a,linear_proxy_delta=qlinear-source_linear,ensemble_proxy_delta=qfull-source_full,feasible=bool(is_feasible)))
   if is_feasible:feasible.append(j)
  if not feasible:raise RuntimeError('Unmodified source must be a feasible fallback')
  j=max(feasible,key=lambda j:cand.loc[j,'alpha']);selected.append(j);znew=control.map_.transform(cand.loc[[j],features])[0]
  dist0=np.sqrt((z0-mu)@prec@(z0-mu)/pdim);distnew=np.sqrt((znew-mu)@prec@(znew-mu)/pdim)
  diagnostics.append(dict(beer=int(beer),selected_alpha=float(cand.loc[j,'alpha']),nontrivial_transfer=bool(cand.loc[j,'alpha']>0),source_target_mahalanobis=dist0,transferred_target_mahalanobis=distnew,target_distance_reduction=1-distnew/dist0,intervention_rms=np.linalg.norm(znew-z0)/np.sqrt(pdim),ensemble_stale_before=source_full,ensemble_stale_after=float(pred.loc[j,'stale_hops']),linear_stale_before=source_linear,linear_stale_after=float(linear.predict(znew[None])[0]),predicted_style=clf.predict(znew[None])[0],target_probability=clf.predict_proba(znew[None])[0,tripel_index]))
 final=cand.loc[selected].reset_index(drop=True);fp,fg=predict_bundle(q1bundle,final,include_grades=True)
 final.to_csv(OUT/'transfer_cross_checked_profiles.csv',index=False);with_ids(final,fp).to_csv(OUT/'transfer_cross_checked_sensory_standardized.csv',index=False);with_ids(final,fg).to_csv(OUT/'transfer_cross_checked_sensory_relative_grades.csv',index=False)
 d=pd.DataFrame(diagnostics);d.to_csv(OUT/'transfer_cross_checked_diagnostics.csv',index=False);pd.DataFrame(allchecks).to_csv(OUT/'transfer_contraction_search.csv',index=False)
 certificates=[]
 for _,row in final.iterrows():
  src=original.loc[row.beer]
  for col,info in DEFECTS.items():
   before=as_raw(src[col],info);after=as_raw(row[col],info);threshold=info['threshold'];observed=np.isfinite(before)
   certificates.append(dict(beer=int(row.beer),chemical=col,unit=info['unit'],source_concentration=before,transferred_concentration=after,source_observed=bool(observed),concentration_delta=after-before if observed else np.nan,threshold=threshold,source_excess=max(before/threshold-1,0) if observed else np.nan,transfer_excess=max(after/threshold-1,0) if np.isfinite(after) else np.nan,certificate='observed concentration nonincrease' if observed else ('no intervention; source remains unobserved' if row.alpha==0 else 'output below threshold; original concentration unknown')))
 pd.DataFrame(certificates).to_csv(OUT/'transfer_cross_checked_chemical_certificate.csv',index=False)
 meta=dict(n_candidates=len(cand),n_sources=len(final),alpha_grid_step=1/80,n_nontrivial=int(d.nontrivial_transfer.sum()),n_full_step=int((d.selected_alpha==1).sum()),n_source_fallback=int((d.selected_alpha==0).sum()),target_distance_reduction_mean=float(d.target_distance_reduction.mean()),intervention_rms_mean=float(d.intervention_rms.mean()),tripel_classifier_count=int((d.predicted_style=='Tripel').sum()),ensemble_stale_max_delta=float((d.ensemble_stale_after-d.ensemble_stale_before).max()),linear_stale_max_delta=float((d.linear_stale_after-d.linear_stale_before).max()),initial_cross_model_failures={s:int((v.delta>1e-8).sum()) for s,v in stalecheck.groupby('method')},interpretation='Agreement between two fitted models trained on the same supplied data; not independent human sensory validation. Grades are Q1 relative calibration, not recovered raw scores.')
 (OUT/'sensory_transfer_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
