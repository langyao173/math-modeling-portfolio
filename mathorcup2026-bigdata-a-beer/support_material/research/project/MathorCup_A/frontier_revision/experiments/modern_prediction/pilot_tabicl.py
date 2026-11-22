"""Timing only: never use this pilot's held-out target values for model choice."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='9'
import time, json, sys, importlib.metadata, resource
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tabicl import TabICLRegressor

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'project'/'MathorCup_A'
sys.path.insert(0,str(BASE/'A_work'/'q1'))
from sensory_model import ChemicalPreprocessor

def main():
    x=pd.read_csv(BASE/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(BASE/'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
    fold=pd.read_csv(BASE/'A_work/q1/fold_assignments.csv').set_index('beer').loc[x.beer,'fold'].to_numpy()
    X=x.iloc[:,3:].to_numpy(float);Y=y.iloc[:,3:].to_numpy(float)
    tr=np.flatnonzero(fold!=0);te=np.flatnonzero(fold==0)
    prep=ChemicalPreprocessor().fit(X[tr]);xr=prep.transform(X[tr]);xt=prep.transform(X[te])
    pc=PCA(64,svd_solver='full').fit(xr)
    views={'raw231':(X[tr],X[te]),'rank_pca64':(pc.transform(xr),pc.transform(xt))}
    torch.set_num_threads(9)
    results=[]
    for name,(a,b) in views.items():
        model=TabICLRegressor(n_estimators=8,n_jobs=9,device='cpu',use_amp=False,
            model_path=HERE/'models/tabicl-regressor-v2-20260212.ckpt',
            allow_auto_download=False,random_state=20261023)
        t=time.perf_counter();model.fit(a,Y[tr,0]);fit=time.perf_counter()-t
        t=time.perf_counter();pred=model.predict(b);predict=time.perf_counter()-t
        row={'view':name,'train':list(a.shape),'test':list(b.shape),'fit_s':fit,'predict_s':predict,
            'n_estimators':8,'threads':9,'finite':bool(np.isfinite(pred).all()),
            'peak_rss_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        results.append(row);print(json.dumps(row),flush=True)
        np.save(HERE/f'pilot_{name}_pred.npy',pred)
        (HERE/'pilot_timing.json').write_text(json.dumps({'results':results,'versions':{
            p:importlib.metadata.version(p) for p in ('tabicl','torch','numpy','scikit-learn')},
            'purpose':'Resource timing only; no target scores calculated or used for model choice.'},indent=2))

if __name__=='__main__':main()
