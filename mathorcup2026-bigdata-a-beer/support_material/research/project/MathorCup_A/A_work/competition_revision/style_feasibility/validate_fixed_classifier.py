"""One fixed-protocol 3-fold diagnostic CV for the C=.1 style classifier.

The fold's ConditionalCopula and rank map are fitted on training rows only.
No hyperparameter search, no modification of the full-data fixed classifier.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;AWORK=HERE.parents[1]
sys.path.insert(0,str(AWORK/'q3'))
from conditional_model import ConditionalCopula
SEED=20261023

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(true,pred):
 tp=int(np.sum((true=='Tripel')&(pred=='Tripel')));fp=int(np.sum((true!='Tripel')&(pred=='Tripel')));fn=int(np.sum((true=='Tripel')&(pred!='Tripel')));tn=int(np.sum((true!='Tripel')&(pred!='Tripel')))
 return dict(n=len(true),accuracy=float(accuracy_score(true,pred)),balanced_accuracy=float(balanced_accuracy_score(true,pred)),Tripel_precision=tp/(tp+fp) if tp+fp else 0.,Tripel_recall=tp/(tp+fn) if tp+fn else 0.,Tripel_TP=tp,Tripel_FP=fp,Tripel_FN=fn,Tripel_TN=tn,Tripel_observed=tp+fn,Tripel_predicted=tp+fp)

def main():
 begin=time.perf_counter();fixed=HERE/'frozen_style_classifier.joblib';before=sha(fixed)
 data=Path(os.environ.get('MATHORCUP_DATA_DIR',AWORK.parent/'data/A')).expanduser().resolve()
 df=pd.read_csv(data/'1.csv');pack=joblib.load(AWORK/'q3/transfer_and_augmentation_models.joblib');features=pack['features'];original=pack['model']
 x=df[features].to_numpy(float);y=df.tasting_category_fine.to_numpy(str);classes=np.sort(np.unique(y));counts=pd.Series(y).value_counts()
 assert len(classes)==22 and int(counts.min())>=3
 protocol=dict(new_diagnostic_validation=True,requested_after_classifier_constraints=True,folds=3,split='StratifiedKFold',shuffle=True,random_state=SEED,
   classifier={'estimator':'LogisticRegression','C':.1,'max_iter':3000,'class_weight':'balanced'},
   control_layer={'class':'ConditionalCopula','kind':original.kind,'kappa':original.kappa,'nu':original.nu},
   preprocessing='Each fold fits all control-layer parameters, empirical normal ranks and missing-value medians on training rows only; validation transformed with that fitted map.',
   hyperparameter_search=False,frozen_full_data_classifier_modified=False,n=len(df),p=len(features),n_classes=len(classes),minimum_class_count=int(counts.min()),
   interpretation='Distinguishability of existing observed styles under this fixed protocol; not causal validation of optimized candidates or a calibrated probability of true physical style.',frozen_classifier_sha256=before)
 (HERE/'classifier_cv_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 oof=np.empty(len(df),dtype=object);proba=np.full((len(df),len(classes)),np.nan);fold_ids=np.full(len(df),-1);folds=[];splits=[];fold_counts=[]
 for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(x,y),1):
  layer=ConditionalCopula(kind=original.kind,kappa=original.kappa,nu=original.nu).fit(x[tr],y[tr])
  zt=layer.map_.transform(x[tr]);zv=layer.map_.transform(x[va]);clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(zt,y[tr])
  oof[va]=clf.predict(zv);fold_ids[va]=fold;proba[va]=clf.predict_proba(zv)[:,[list(clf.classes_).index(c) for c in classes]]
  folds.append(dict(fold=fold,n_train=len(tr),**metrics(y[va],oof[va])))
  for idx in tr:splits.append(dict(fold=fold,split='train',beer=int(df.beer.iloc[idx]),style=y[idx]))
  for idx in va:splits.append(dict(fold=fold,split='validation',beer=int(df.beer.iloc[idx]),style=y[idx]))
  for c in classes:fold_counts.append(dict(fold=fold,style=c,n_train=int(np.sum(y[tr]==c)),n_validation=int(np.sum(y[va]==c))))
  print(f'fold {fold}: {json.dumps(folds[-1])}',flush=True)
 assert np.all(fold_ids>0) and np.isfinite(proba).all()
 result=pd.DataFrame({'beer':df.beer,'style_observed':y,'style_oof_predicted':oof,'fold':fold_ids,'Tripel_oof_probability':proba[:,list(classes).index('Tripel')]})
 for j,c in enumerate(classes):result['p_'+c]=proba[:,j]
 result.to_csv(HERE/'classifier_cv_oof_predictions.csv',index=False)
 pd.DataFrame(folds).to_csv(HERE/'classifier_cv_fold_metrics.csv',index=False);pd.DataFrame(splits).to_csv(HERE/'classifier_cv_splits.csv',index=False);pd.DataFrame(fold_counts).to_csv(HERE/'classifier_cv_class_counts.csv',index=False)
 cm=confusion_matrix(y,oof,labels=classes);pd.DataFrame(cm,index=pd.Index(classes,name='observed_style'),columns=classes).to_csv(HERE/'classifier_cv_confusion.csv')
 pd.DataFrame(cm/cm.sum(axis=1)[:,None],index=pd.Index(classes,name='observed_style'),columns=classes).to_csv(HERE/'classifier_cv_confusion_row_fraction.csv')
 prec,rec,f1,support=precision_recall_fscore_support(y,oof,labels=classes,zero_division=0)
 pd.DataFrame({'style':classes,'precision':prec,'recall':rec,'f1':f1,'support':support,'predicted_count':cm.sum(axis=0)}).to_csv(HERE/'classifier_cv_class_metrics.csv',index=False)
 aggregate=metrics(y,oof);aggregate.update(protocol='one 3-fold stratified CV, all preprocessing learned within training fold',seed=SEED,n_classes=len(classes),class_counts=counts.sort_index().to_dict(),
   full_classifier_sha256_unchanged=sha(fixed)==before,seconds=time.perf_counter()-begin)
 pd.DataFrame([[aggregate['Tripel_TN'],aggregate['Tripel_FP']],[aggregate['Tripel_FN'],aggregate['Tripel_TP']]],index=['observed_non_Tripel','observed_Tripel'],columns=['predicted_non_Tripel','predicted_Tripel']).to_csv(HERE/'classifier_cv_Tripel_confusion.csv')
 (HERE/'classifier_cv_summary.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2))
 # Independent integer accounting verifies the summary from the saved confusion.
 saved=pd.read_csv(HERE/'classifier_cv_confusion.csv',index_col=0);acc=np.trace(saved.to_numpy())/saved.to_numpy().sum();bal=np.mean(np.diag(saved.to_numpy())/saved.sum(axis=1).to_numpy())
 tp=int(saved.loc['Tripel','Tripel']);precision=tp/saved['Tripel'].sum();recall=tp/saved.loc['Tripel'].sum()
 check=dict(confusion_total=int(saved.to_numpy().sum()),accuracy_abs_difference=float(abs(acc-aggregate['accuracy'])),balanced_accuracy_abs_difference=float(abs(bal-aggregate['balanced_accuracy'])),Tripel_precision_abs_difference=float(abs(precision-aggregate['Tripel_precision'])),Tripel_recall_abs_difference=float(abs(recall-aggregate['Tripel_recall'])),n_unique_validation_beers=int(result.beer.nunique()),each_beer_has_one_validation_prediction=bool(result.beer.nunique()==len(df)),full_classifier_unchanged=sha(fixed)==before)
 assert all(check[k]<1e-12 for k in check if k.endswith('_abs_difference')) and check['each_beer_has_one_validation_prediction'] and check['full_classifier_unchanged']
 (HERE/'classifier_cv_audit.json').write_text(json.dumps(check,ensure_ascii=False,indent=2));print(json.dumps(aggregate,ensure_ascii=False,indent=2))

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
