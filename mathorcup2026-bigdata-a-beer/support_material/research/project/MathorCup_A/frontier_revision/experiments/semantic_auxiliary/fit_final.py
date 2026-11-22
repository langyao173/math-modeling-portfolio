"""Fit a deployable training-only semantic correction; no semantic input API."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from run_semantic_rrr import HERE,AW,PROJECT,SEED,SPECS,grid_predictions,SemanticRRR
from sensory_model import predict_bundle,grades_from_reference

def predict_semantic_bundle(bundle,chemistry,include_grades=False):
    if isinstance(bundle,(str,Path)):bundle=joblib.load(bundle)
    X=chemistry[bundle['features']].to_numpy(float) if isinstance(chemistry,pd.DataFrame) else np.asarray(chemistry,float)
    old=predict_bundle(bundle['baseline'],X).to_numpy(float)
    w=bundle['weight'];pred=(1-w)*old+w*bundle['semantic'].predict(X)
    frame=pd.DataFrame(pred,columns=bundle['targets'])
    if include_grades:return frame,pd.DataFrame(grades_from_reference(pred,bundle['reference_y']),columns=bundle['targets'])
    return frame

def main():
    xx=pd.read_csv(PROJECT/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    yy=pd.read_csv(PROJECT/'data/A/4.csv').set_index('beer').loc[xx.beer]
    ss=pd.read_csv(PROJECT/'data/A/7.csv').set_index('beer').loc[xx.beer]
    targets=yy.columns[2:].tolist();features=xx.columns[3:].tolist()
    X=xx[features].to_numpy(float);Y=yy[targets].to_numpy(float);S=ss.iloc[:,2:].to_numpy(float)
    scores=np.zeros(len(SPECS));folds=KFold(5,shuffle=True,random_state=SEED)
    for tr,te in folds.split(X):
        p=grid_predictions(X[tr],Y[tr],S[tr],X[te])
        sd=np.maximum(Y[tr].std(0,ddof=1),1e-10)
        scores+=np.mean(((p-Y[te][None])/sd)**2,axis=(1,2))/5
    allowed=np.array([i for i,s in enumerate(SPECS) if s['tau']>0])
    best=int(allowed[np.argmin(scores[allowed])]);spec=SPECS[best]
    model=SemanticRRR(spec).fit(X,Y,S)
    # Train-side cross-fitted residuals determine the final deployment weight.
    aux=np.load(HERE/'oof_predictions.npz')['aligned_aux_all50']
    old=np.load(AW/'q1/oof_arrays.npz')['nested_ensemble']
    sd=np.maximum(Y.std(0,ddof=1),1e-10)
    residual=(old-Y)/sd;change=(aux-old)/sd
    w=float(np.clip(-np.mean(residual*change)/(np.mean(change**2)+.02),0,1))
    base=joblib.load(AW/'q1/sensory_ensemble.joblib')
    bundle=dict(features=features,targets=targets,baseline=base,semantic=model,
                weight=w,reference_y=Y,spec=spec,
                input_contract='Only chemistry attachment 1; auxiliary review words were training targets.',
                evaluation_contract='Use blend_oof.npz for held-out performance, never fitted predictions here.')
    joblib.dump(bundle,HERE/'semantic_corrected_predictor.joblib')
    p,g=predict_semantic_bundle(bundle,xx,True)
    p.insert(0,'beer',xx.beer.to_numpy());g.insert(0,'beer',xx.beer.to_numpy())
    p.to_csv(HERE/'all_beers_fitted_continuous.csv',index=False)
    g.to_csv(HERE/'all_beers_fitted_relative_grades.csv',index=False)
    # Rowwise and batch inference agree without any auxiliary observations.
    np.testing.assert_allclose(p.iloc[:7,1:].to_numpy(),np.vstack([
        predict_semantic_bundle(bundle,X[i:i+1]).to_numpy() for i in range(7)]),atol=1e-10,rtol=1e-10)
    report=dict(spec=spec,final_weight=w,full_data_cv_nmse=float(scores[best]),
                n=250,p=231,q=50,semantic_inference_inputs=0,
                output_interpretation='Fitted outputs for user application; held-out accuracy stored separately.',
                batch_rowwise_equivalence=True)
    (HERE/'final_fit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
