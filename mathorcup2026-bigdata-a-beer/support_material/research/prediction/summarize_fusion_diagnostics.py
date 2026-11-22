"""Descriptive task outputs for the fixed, already completed fusion experiment."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=ROOT/'fusion_replay_v1')
parser.add_argument('--project-dir',type=Path,default=ROOT.parent/'project/MathorCup_A')
args=parser.parse_args()
OUT=args.results
PROJECT=args.project_dir
with np.load(OUT/'outer_predictions.npz',allow_pickle=False) as z:
    a={k:z[k] for k in z.files}
Y,fold=a['Y'],a['fold'];variance=Y.var(0)
x=pd.read_csv(PROJECT/'data/A/1.csv').set_index('beer').loc[a['beer']]
methods={k:v for k,v in a.items() if k not in ['Y','beer','fold','targets']}
rows=[];dim=[];sty=[]
truth_grade=np.empty(Y.shape,int)
for f in range(5):
    tr,te=fold!=f,fold==f
    cut=np.quantile(Y[tr],np.arange(1,7)/7,axis=0)
    for j in range(50):truth_grade[te,j]=np.searchsorted(cut[:,j],Y[te,j],side='right')
for n,p in methods.items():
    pg=np.empty(Y.shape,int)
    for f in range(5):
        tr,te=fold!=f,fold==f
        cut=np.quantile(Y[tr],np.arange(1,7)/7,axis=0)
        for j in range(50):pg[te,j]=np.searchsorted(cut[:,j],p[te,j],side='right')
    err=np.abs(pg-truth_grade)
    loss=(Y-p)**2
    for g,js in [('all50',np.arange(50)),('core30',np.arange(30)),('other20',np.arange(30,50))]:
        rows.append({'method':n,'targets':g,'grade_mae':float(err[:,js].mean()),'grade_exact':float((err[:,js]==0).mean()),'grade_within_one':float((err[:,js]<=1).mean())})
        for s in sorted(x.tasting_category_fine.unique()):
            ix=x.tasting_category_fine.eq(s).to_numpy();sty.append({'method':n,'targets':g,'style':s,'n':int(ix.sum()),'nmse':float((loss[ix][:,js]/variance[js]).mean())})
    for j,t in enumerate(a['targets']):dim.append({'method':n,'target':t,'core30':j<30,'r2':float(1-loss[:,j].mean()/variance[j]),'rmse':float(np.sqrt(loss[:,j].mean()))})
pd.DataFrame(rows).to_csv(OUT/'relative_grade_metrics.csv',index=False)
pd.DataFrame(dim).to_csv(OUT/'dimension_metrics.csv',index=False)
sty=pd.DataFrame(sty);sty.to_csv(OUT/'style_metrics.csv',index=False)
pairs=[]
for g in ['all50','core30','other20']:
    ours=sty.query('method == "inner_groupwise_y16_native50" and targets == @g').set_index('style')
    for ref in ['raw231_y16_no_residual','raw231_y50_native','old_nested_ensemble','tabicl_nested_selection']:
        theirs=sty.query('method == @ref and targets == @g').set_index('style')
        d=ours.nmse-theirs.nmse
        for threshold in [0,10]:
            mask=ours.n>=threshold
            pairs.append({'targets':g,'reference':ref,'minimum_style_n':threshold,'style_groups':int(mask.sum()),'groupwise_lower_nmse_groups':int((d[mask]<0).sum()),'style_equal_delta_nmse':float(d[mask].mean())})
pd.DataFrame(pairs).to_csv(OUT/'style_pairwise_summary.csv',index=False)
print(pd.DataFrame(rows).query('method == "inner_groupwise_y16_native50"').to_string(index=False))
print(pd.DataFrame(pairs).query('targets == "core30"').to_string(index=False))
