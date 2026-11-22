"""Diagnostic-triggered refinements, with full nested internal validation.

The original locked test already exposed a flaw in the smooth copula. This file
therefore reports INTERNAL NESTED VALIDATION, never a new external/untouched test.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,joblib
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score
from conditional_model import ConditionalCopula,IntervalCopula,DERIVED,TECHNICAL
from run_q3 import score_model,paired_ci,SEED,DEFAULT_DATA

OUT=Path(__file__).parent
PARAMS=[
 dict(name='smooth_pooled',kind='pooled',kappa=2,nu=20),
 dict(name='interval_gaussian',kind='interval_gaussian',kappa=2,nu=20),
 dict(name='interval_kernel_h03',kind='interval_kernel',kappa=2,nu=20,bandwidth=.3),
 dict(name='interval_kernel_h06',kind='interval_kernel',kappa=2,nu=20,bandwidth=.6),
 dict(name='interval_kernel_h09',kind='interval_kernel',kappa=2,nu=20,bandwidth=.9),
]

def fit(p,x,s):
 cls=IntervalCopula if p['kind'].startswith('interval') else ConditionalCopula
 return cls(**{k:v for k,v in p.items() if k!='name'}).fit(x,s)

def evaluate(m,x,s,seed=SEED):
 es,vs,z,zg,sg=score_model(m,x,s,n_sample=256,seed=seed)
 combined=np.vstack([z,zg]);isgen=np.r_[np.zeros(len(z)),np.ones(len(zg))]
 disc=RandomForestClassifier(n_estimators=120,max_depth=4,min_samples_leaf=4,n_jobs=1,random_state=seed)
 pr=cross_val_predict(disc,combined,isgen,method='predict_proba',cv=StratifiedKFold(4,shuffle=True,random_state=seed))[:,1]
 auc=roc_auc_score(isgen,pr)
 # The discriminator could invert labels: use distinguishability >= .5.
 distinguishability=max(auc,1-auc)
 real_nn=cdist(z,m.z_).min(axis=1)/np.sqrt(z.shape[1]);gen_nn=cdist(zg,m.z_).min(axis=1)/np.sqrt(z.shape[1])
 ratio=np.median(gen_nn)/np.median(real_nn)
 selection_score=es.mean()+.15*max(0,distinguishability-.65)+.05*max(0,.5-ratio)
 clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(m.z_,m.style_)
 record=dict(energy=es.mean(),variogram=vs.mean(),c2st_auc=auc,distinguishability=distinguishability,nn_ratio=ratio,exact_copy_rate=np.mean(gen_nn<1e-8),real_style_accuracy=accuracy_score(s,clf.predict(z)),real_style_balanced_accuracy=balanced_accuracy_score(s,clf.predict(z)),generated_style_accuracy=accuracy_score(sg,clf.predict(zg)),selection_score=selection_score)
 return record,es,vs

def main():
 df=pd.read_csv(DEFAULT_DATA/'1.csv');features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL]
 x=df[features].to_numpy(float);s=df.tasting_category_fine.to_numpy(str)
 outer=StratifiedKFold(3,shuffle=True,random_state=SEED+300)
 inner_rows=[];outer_rows=[];beer_rows=[];chosen=[]
 for fold,(a,b) in enumerate(outer.split(x,s)):
  inner=StratifiedKFold(2,shuffle=True,random_state=SEED+400+fold)
  local=[]
  for infold,(ia,ib) in enumerate(inner.split(x[a],s[a])):
   aa,bb=a[ia],a[ib]
   for p in PARAMS:
    m=fit(p,x[aa],s[aa]);record,*_=evaluate(m,x[bb],s[bb],SEED+fold*10+infold)
    row=dict(outer_fold=fold,inner_fold=infold,model=p['name'],**record);inner_rows.append(row);local.append(row)
  summary=pd.DataFrame(local).groupby('model').selection_score.mean()
  selected=summary.idxmin();chosen.append(selected)
  print('Outer fold',fold,'inner selected',selected,flush=True)
  for p in PARAMS:
   m=fit(p,x[a],s[a]);record,es,vs=evaluate(m,x[b],s[b],SEED+500+fold)
   outer_rows.append(dict(outer_fold=fold,model=p['name'],selected=(p['name']==selected),n_validation=len(b),**record))
   for j,idx in enumerate(b):
    beer_rows.append(dict(outer_fold=fold,model=p['name'],selected=(p['name']==selected),beer=int(df.beer.iloc[idx]),style=s[idx],energy=es[j],variogram=vs[j]))
  print('Outer fold',fold,'completed',flush=True)
 pd.DataFrame(inner_rows).to_csv(OUT/'extension_inner_selection.csv',index=False)
 pd.DataFrame(outer_rows).to_csv(OUT/'extension_outer_comparison.csv',index=False)
 bs=pd.DataFrame(beer_rows);bs.to_csv(OUT/'extension_outer_scores_by_beer.csv',index=False)
 outerdf=pd.DataFrame(outer_rows)
 summary=outerdf.groupby('model').mean(numeric_only=True)
 summary.to_csv(OUT/'extension_nested_model_summary.csv')
 selected_rows=outerdf[outerdf.selected]
 (OUT/'extension_selection_process.json').write_text(json.dumps(dict(validation_type='Post-diagnostic nested INTERNAL validation; no new independent external test data.',outer_folds=3,inner_folds=2,selected_each_outer=chosen,selection_rule='energy + 0.15 max(0,max(AUC,1-AUC)-0.65) + 0.05 max(0,0.5 - median_generated_NN/median_real_NN)',selected_nested_energy=float(bs[bs.selected].energy.mean()),selected_nested_variogram=float(bs[bs.selected].variogram.mean()),selected_nested_c2st_auc=float(selected_rows.c2st_auc.mean()),selected_nested_nn_ratio=float(selected_rows.nn_ratio.mean()),selected_nested_copy_rate=float(selected_rows.exact_copy_rate.mean())),ensure_ascii=False,indent=2))
 # Final choice uses all inner-selection scores, not outer results.
 final_choice=pd.DataFrame(inner_rows).groupby('model').selection_score.mean().idxmin()
 p=next(p for p in PARAMS if p['name']==final_choice)
 model=fit(p,x,s)
 joblib.dump(dict(model=model,features=features,all_features=list(df.columns[3:]),selected_parameter=p),OUT/'revised_conditional_generator.joblib')
 rng=np.random.default_rng(SEED+600);rows=[];prototypes=[]
 for st in model.styles_:
  g,_=model.sample(st,100,rng);f=pd.DataFrame(g,columns=features)
  f.insert(0,'tasting_category_fine',st);f.insert(0,'synthetic_id',[st.replace('/','_').replace(' ','_')+f'_{j+1:03d}' for j in range(len(g))]);rows.append(f)
  prototypes.append(dict(tasting_category_fine=st,**dict(zip(features,model.prototype(st)))))
 generated=pd.concat(rows,ignore_index=True)
 proto=pd.DataFrame(prototypes)
 for tech in TECHNICAL:
  generated[tech]=df[tech].median();proto[tech]=df[tech].median()
 generated.to_csv(OUT/'revised_generated_chemical_profiles.csv',index=False)
 proto.to_csv(OUT/'revised_style_prototypes.csv',index=False)
 (OUT/'revised_selected_model.json').write_text(json.dumps(p,indent=2))
 print('Final revised model',final_choice,flush=True)

if __name__=='__main__':main()
