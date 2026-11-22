"""Lag-aligned accounting features; no record after the scoring month is accessed."""
import numpy as np,pandas as pd,warnings
CASH=['AMT_BALANCE','AMT_DRAWINGS_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','AMT_TOTAL_RECEIVABLE','AMT_RECEIVABLE_PRINCIPAL','AMT_PAYMENT_CURRENT','AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_POS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL','SK_DPD','SK_DPD_DEF']
def augment(raw,snapshots):
 from feature_engine import keys
 s=snapshots.reset_index(drop=True);d=raw[raw.SK_ID_PREV.isin(s.account_id.unique())].reset_index(drop=True)
 idx=pd.Index(keys(d.SK_ID_PREV,d.MONTHS_BALANCE));data=d[CASH].to_numpy(float);a=np.full((len(s),12,len(CASH)),np.nan)
 for lag in range(12):
  j=idx.get_indexer(keys(s.account_id,s.anchor_month-lag));good=j>=0;a[good,lag]=data[j[good]]
 out={}
 for lag in (2,3,4,5):
  for j,c in enumerate(CASH):out[f'extra_lag{lag}_{c}']=a[:,lag,j]
 def v(c):return a[:,:,CASH.index(c)]
 b,p,m,dw,l,rec,prin=[v(c) for c in ['AMT_BALANCE','AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','AMT_DRAWINGS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL','AMT_TOTAL_RECEIVABLE','AMT_RECEIVABLE_PRINCIPAL']]
 series={'payment_prior_balance':p[:,:11]/np.maximum(abs(b[:,1:]),1),'payment_prior_minimum':p[:,:11]/np.maximum(abs(m[:,1:]),1),'net_paydown_prior_balance':(p[:,:11]-dw[:,:11])/np.maximum(abs(b[:,1:]),1),'unreconciled_balance':(b[:,:11]-b[:,1:]-dw[:,:11]+p[:,:11])/np.maximum(abs(l[:,1:]),1),'charges_share':(rec-prin)/np.maximum(abs(rec),1),'scheduled_burden':m/np.maximum(abs(b),1),'nonpayment':np.where(np.isfinite(p)&np.isfinite(m),((p<=0)&(m>0)).astype(float),np.nan),'shortfall_prior_minimum':np.maximum(m[:,1:]-p[:,:11],0)/np.maximum(abs(m[:,1:]),1)}
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for name,z in series.items():
   out['extra_'+name+'_current']=z[:,0]
   for w in (3,6,11):
    x=z[:,:w];out[f'extra_{name}_mean{w}']=np.nanmean(x,axis=1);out[f'extra_{name}_std{w}']=np.nanstd(x,axis=1);out[f'extra_{name}_min{w}']=np.nanmin(x,axis=1)
  for name,z in [('balance',b),('payment',p),('minimum',m),('drawings',dw),('charges',rec-prin),('limit',l)]:
   scale=np.maximum(np.nanmean(abs(z[:,:6]),axis=1),1)
   for w in (3,6,12):
    x=z[:,:w];present=np.isfinite(x);t=np.broadcast_to(-np.arange(w,dtype=float),x.shape);tm=np.where(present,t,np.nan);xm=np.nanmean(x,axis=1);mt=np.nanmean(tm,axis=1)
    slope=np.nansum((x-xm[:,None])*(tm-mt[:,None]),axis=1)/np.maximum(np.nansum((tm-mt[:,None])**2,axis=1),1)
    out[f'extra_{name}_slope{w}']=slope/scale;out[f'extra_{name}_coverage{w}']=present.mean(axis=1)
  for w in (3,6,12):
   delta=(l[:,:w-1]-l[:,1:w])/np.maximum(abs(l[:,1:w]),1);out[f'extra_limit_increases{w}']=np.nansum(delta>.01,axis=1);out[f'extra_limit_decreases{w}']=np.nansum(delta<-.01,axis=1)
 return pd.DataFrame(out).replace([np.inf,-np.inf],np.nan).astype('float32')
if __name__=='__main__':
 from pathlib import Path
 import sys,json,gc
 root=Path(__file__).resolve().parents[1];cache=Path(sys.argv[1]);source=Path(sys.argv[2]);sys.path.insert(0,str(source/'code'));raw=pd.read_pickle(cache/'raw.pkl')
 for name in ['development','evaluation']:
  f=pd.read_pickle(cache/f'{name}_features.pkl');xs=[augment(raw,f.iloc[j:j+4000]) for j in range(0,len(f),4000)];a=pd.concat(xs,ignore_index=True);a.to_pickle(cache/f'{name}_extra.pkl');(root/'data/extra_columns.json').write_text(json.dumps(a.columns.tolist(),indent=2));print(name,a.shape,flush=True);del a,xs,f;gc.collect()
