"""Reproduce Task 1 nested validation, missingness tests and predictions.
No true raw 0-6 labels are present: integer outputs are declared relative grades.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import joblib
from sensory_model import SensoryEstimator,specifications,learn_weights,grades_from_reference

SEED=20261023
FAMILIES=['ridge_raw','ridge_rank','rbf','pls','extra_trees','random_forest']

def tune(X,Y,seed):
    inner=list(KFold(3,shuffle=True,random_state=seed).split(X))
    candidates={}
    rows=[]
    for spec in specifications():
        pp=np.empty_like(Y)
        for tr,te in inner:
            pp[te]=SensoryEstimator(spec).fit(X[tr],Y[tr]).predict(X[te])
        score=float(np.mean(((pp-Y)/np.std(Y,axis=0,ddof=1))**2))
        f=spec['family'];rows.append({'spec':spec,'inner_nmse':score})
        if f not in candidates or score<candidates[f][0]:
            candidates[f]=(score,spec,pp)
    w=learn_weights([candidates[f][2] for f in FAMILIES],Y)
    return candidates,w,rows

def assess(y,p):
    r=r2_score(y,p,multioutput='raw_values')
    rmse=np.sqrt(np.mean((p-y)**2,axis=0))
    return dict(macro_r2=float(np.mean(r)),median_r2=float(np.median(r)),
        positive_r2=int((r>0).sum()),macro_rmse=float(np.mean(rmse)),
        macro_spearman=float(np.nanmean([spearmanr(y[:,j],p[:,j]).statistic for j in range(y.shape[1])])),
        r2=r,rmse=rmse)

def jsonable(r):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    base=Path(args.data)
    x=pd.read_csv(base/'1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(base/'4.csv').set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.beer,y.beer)
    features=x.columns[3:].tolist();targets=y.columns[3:].tolist()
    X=x[features].to_numpy(float);Y=y[targets].to_numpy(float)
    folds=list(KFold(5,shuffle=True,random_state=SEED).split(X))
    fold_id=np.empty(len(X),int)
    predictions={f:np.empty_like(Y) for f in FAMILIES+['nested_ensemble','training_mean','style_mean']}
    masked={str(r):np.empty_like(Y) for r in [0,.1,.3,.5]}
    masked_knn={str(r):np.empty_like(Y) for r in [.1,.3,.5]}
    block_masks={k:np.empty_like(Y) for k in ['first37_unavailable','gcms_unavailable','high_concentration_missing']}
    grades=np.empty_like(Y,dtype=int);grade_true=np.empty_like(grades)
    alternatives={k:np.empty_like(grades) for k in ['minmax','q05q95']}
    lows=np.empty_like(Y);highs=np.empty_like(Y)
    selected=[];fold_metrics=[]
    start=time.time()
    for fold,(tr,te) in enumerate(folds):
        fold_id[te]=fold
        print(f'Outer {fold+1}/5 start; elapsed {time.time()-start:.1f}s',flush=True)
        cand,weights,tuning=tune(X[tr],Y[tr],SEED+fold)
        models=[SensoryEstimator(cand[f][1]).fit(X[tr],Y[tr]) for f in FAMILIES]
        knn_models=[SensoryEstimator(cand[f][1],imputer='knn').fit(X[tr],Y[tr]) for f in FAMILIES]
        pp=[m.predict(X[te]) for m in models]
        for f,p in zip(FAMILIES,pp):predictions[f][te]=p
        pe=sum(w*p for w,p in zip(weights,pp));predictions['nested_ensemble'][te]=pe
        predictions['training_mean'][te]=np.mean(Y[tr],axis=0)
        train_styles=x.iloc[tr]['tasting_category_fine'].to_numpy()
        for i in te:
            match=train_styles==x.iloc[i]['tasting_category_fine']
            predictions['style_mean'][i]=np.mean(Y[tr][match],axis=0) if match.any() else np.mean(Y[tr],axis=0)
        grades[te]=grades_from_reference(pe,Y[tr]);grade_true[te]=grades_from_reference(Y[te],Y[tr])
        for key,q in [('minmax',[0,1]),('q05q95',[.05,.95])]:
            a,b=np.quantile(Y[tr],q,axis=0)
            alternatives[key][te]=np.floor(np.clip(6*(pe-a)/np.maximum(b-a,1e-8),0,6)+.5).astype(int)
        inner_p=sum(w*cand[f][2] for w,f in zip(weights,FAMILIES))
        # Diagnostic cross-conformal bands; adaptive fitting prevents a formal finite-sample claim.
        resid=np.abs(Y[tr]-inner_p)
        q=np.quantile(resid,.9,axis=0,method='higher')
        lows[te]=pe-q;highs[te]=pe+q
        masked['0'][te]=pe
        for rate in [.1,.3,.5]:
            rng=np.random.default_rng(SEED+1000*fold+int(rate*100))
            xm=X[te].copy();xm[rng.random(xm.shape)<rate]=np.nan
            masked[str(rate)][te]=sum(w*m.predict(xm) for w,m in zip(weights,models))
            masked_knn[str(rate)][te]=sum(w*m.predict(xm) for w,m in zip(weights,knn_models))
        for key in block_masks:
            xm=X[te].copy()
            if key=='first37_unavailable':xm[:,:37]=np.nan
            elif key=='gcms_unavailable':xm[:,37:-4]=np.nan
            else:
                thresholds=np.nanquantile(X[tr],.9,axis=0)
                xm[xm>thresholds]=np.nan
            block_masks[key][te]=sum(w*m.predict(xm) for w,m in zip(weights,models))
        selected.append({'fold':fold,'train_beer':x.iloc[tr].beer.tolist(),'test_beer':x.iloc[te].beer.tolist(),
                         'specs':{f:cand[f][1] for f in FAMILIES},'weights':dict(zip(FAMILIES,weights.tolist())),
                         'tuning':tuning})
        fold_metrics.append({'fold':fold,**{k:v for k,v in assess(Y[te],pe).items() if k not in ['r2','rmse']}})
        joblib.dump({'features':features,'targets':targets,'models':models,'weights':weights,
                     'reference_y':Y[tr]},out/f'fold_{fold}_models.joblib')
        print(f'Outer {fold+1}/5 done, ensemble R2={fold_metrics[-1]["macro_r2"]:.4f}',flush=True)
    scores={f:jsonable(assess(Y,p)) for f,p in predictions.items()}
    pd.DataFrame([{k:v for k,v in dict(model=f,**s).items() if k not in ['r2','rmse']} for f,s in scores.items()]).to_csv(out/'model_comparison.csv',index=False)
    rows=[]
    for f,s in scores.items():
        for j,t in enumerate(targets):rows.append({'model':f,'target':t,'r2':s['r2'][j],'rmse':s['rmse'][j]})
    pd.DataFrame(rows).to_csv(out/'dimension_metrics.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
    meta=x[['beer','beer_id','tasting_category_fine']].copy();meta['fold']=fold_id
    pd.concat([meta,pd.DataFrame(predictions['nested_ensemble'],columns=targets)],axis=1).to_csv(out/'oof_predictions_standardized.csv',index=False)
    pd.concat([meta,pd.DataFrame(grades,columns=targets)],axis=1).to_csv(out/'oof_relative_grades_0_6.csv',index=False)
    pd.concat([meta,pd.DataFrame(lows,columns=targets)],axis=1).to_csv(out/'oof_interval_lower90.csv',index=False)
    pd.concat([meta,pd.DataFrame(highs,columns=targets)],axis=1).to_csv(out/'oof_interval_upper90.csv',index=False)
    pd.DataFrame({'beer':x.beer,'fold':fold_id}).to_csv(out/'fold_assignments.csv',index=False)
    masking=[]
    for imp,dd in [('median',masked),('knn',masked_knn)]:
        for rate,p in dd.items():
            masking.append({'scenario':'MCAR','imputer':imp,'missing_fraction':float(rate),**{k:v for k,v in assess(Y,p).items() if k not in ['r2','rmse']}})
    for name,p in block_masks.items():masking.append({'scenario':name,'imputer':'median','missing_fraction':np.nan,**{k:v for k,v in assess(Y,p).items() if k not in ['r2','rmse']}})
    pd.DataFrame(masking).to_csv(out/'missingness_stress.csv',index=False)
    pd.DataFrame({'target':targets,'relative_grade_accuracy':(grades==grade_true).mean(axis=0),
                  'relative_grade_mae':np.abs(grades-grade_true).mean(axis=0),
                  'within_one_grade':(np.abs(grades-grade_true)<=1).mean(axis=0),
                  'interval_coverage':((Y>=lows)&(Y<=highs)).mean(axis=0),
                  'interval_width':(highs-lows).mean(axis=0),
                  'minmax_disagreement':(grades!=alternatives['minmax']).mean(axis=0),
                  'q05q95_disagreement':(grades!=alternatives['q05q95']).mean(axis=0)}).to_csv(out/'grade_and_interval_diagnostics.csv',index=False)
    # Paired beer bootstrap of fixed OOF predictions: descriptive sampling uncertainty.
    rng=np.random.default_rng(SEED);boot=[]
    for b in range(1000):
        ii=rng.integers(len(Y),size=len(Y))
        yy=Y[ii]; rr={f:float(np.mean(r2_score(yy,p[ii],multioutput='raw_values'))) for f,p in predictions.items()}
        boot.append(rr)
    bb=pd.DataFrame(boot)
    ci={f:{'lower':float(bb[f].quantile(.025)),'upper':float(bb[f].quantile(.975))} for f in bb}
    pair={f:{'delta':scores['nested_ensemble']['macro_r2']-scores[f]['macro_r2'],
              'lower':float((bb.nested_ensemble-bb[f]).quantile(.025)),
              'upper':float((bb.nested_ensemble-bb[f]).quantile(.975))} for f in FAMILIES}
    # Separate 5-fold generalization to held-out styles, using one fixed robust family.
    style_pred=np.empty_like(Y)
    groups=x.tasting_category_fine.to_numpy()
    for tr,te in GroupKFold(5).split(X,Y,groups):
        model=SensoryEstimator(dict(family='random_forest',mode='raw',max_features=.5,min_samples_leaf=2)).fit(X[tr],Y[tr])
        style_pred[te]=model.predict(X[te])
    style_result=jsonable(assess(Y,style_pred))
    pd.concat([meta,pd.DataFrame(style_pred,columns=targets)],axis=1).to_csv(out/'heldout_style_predictions.csv',index=False)
    print('Final nested tuning and fit',flush=True)
    cand,weights,tuning=tune(X,Y,SEED+99)
    models=[SensoryEstimator(cand[f][1]).fit(X,Y) for f in FAMILIES]
    bundle={'features':features,'targets':targets,'models':models,'weights':weights,'reference_y':Y,
            'families':FAMILIES,'specs':[cand[f][1] for f in FAMILIES],
            'grade_interpretation':'Relative within-study intensity grades; not recovered raw panel scores.'}
    joblib.dump(bundle,out/'sensory_ensemble.joblib')
    summary={'seed':SEED,'n':len(X),'p':len(features),'targets':len(targets),
             'models':scores,'bootstrap_ci':ci,'paired_bootstrap':pair,
             'heldout_style_fixed_rf':style_result,
             'relative_grades':{'exact':float(np.mean(grades==grade_true)),
                 'within_one':float(np.mean(np.abs(grades-grade_true)<=1)),
                 'mae':float(np.mean(np.abs(grades-grade_true))),
                 'minmax_disagreement':float(np.mean(grades!=alternatives['minmax'])),
                 'q05q95_disagreement':float(np.mean(grades!=alternatives['q05q95']))},
             'intervals':{'empirical_coverage':float(np.mean((Y>=lows)&(Y<=highs))),
                 'mean_width':float(np.mean(highs-lows)),
                 'interpretation':'Nested cross-conformal diagnostic; no exact split-conformal guarantee.'},
             'final_weights':dict(zip(FAMILIES,weights.tolist())),
             'wall_seconds':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'nested_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    (out/'final_selection.json').write_text(json.dumps(tuning,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'oof_arrays.npz',Y=Y,**predictions)
    print(json.dumps({'model_comparison':[{k:v for k,v in dict(model=f,**s).items() if k not in ['r2','rmse']} for f,s in scores.items()],
                     'weights':summary['final_weights'],'elapsed':summary['wall_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=os.environ.get('MATHORCUP_DATA_DIR','/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))
    p.add_argument('--output',default=str(Path(__file__).parent))
    main(p.parse_args())
