"""Full-history reference transformation; preserves actual behavior and missing slots."""
import hashlib,warnings
import numpy as np
import pandas as pd
NUM=['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
def keys(a,t):return np.asarray(a,dtype='int64')*128+np.asarray(t,dtype='int64')+100
def features(d,s,reference_limits=None,current_limits=None):
 s=s.copy().reset_index(drop=True)
 idx=pd.Index(keys(d.SK_ID_PREV,d.MONTHS_BALANCE));n=len(s)
 cols=NUM+['SK_DPD','SK_DPD_DEF'];raw=d[cols].to_numpy(float)
 frames=[];presence=[];out={}
 for lag in range(12):
  ix=idx.get_indexer(keys(s.account_id,s.anchor_month-lag));a=np.full((n,len(cols)),np.nan);good=ix>=0;a[good]=raw[ix[good]]
  if reference_limits is not None:
   a[good,cols.index('AMT_CREDIT_LIMIT_ACTUAL')]=np.asarray(reference_limits)[good]
  elif current_limits is not None and lag==0:
   a[good,cols.index('AMT_CREDIT_LIMIT_ACTUAL')]=np.asarray(current_limits)[good]
  frames.append(a);presence.append(good)
  if lag<2:
   pre=['anchor','lag1'][lag]
   for j,c in enumerate(cols):out[pre+'__'+c]=a[:,j]
 stack=np.stack(frames,axis=1);present=np.stack(presence,axis=1)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for w in (3,6,12):
   for j,c in enumerate(cols):
    if w==6 and c in NUM:continue
    out[f'w{w}_mean__{c}']=np.nanmean(stack[:,:w,j],axis=1)
    out[f'w{w}_max__{c}']=np.nanmax(stack[:,:w,j],axis=1)
   for th in (1,7,30):
    cnt=np.sum(stack[:,:w,-2]>=th,axis=1).astype(float)
    out[f'dpd{th}_w{w}_count']=cnt;out[f'dpd{th}_w{w}_share']=cnt/present[:,:w].sum(axis=1)
 ix=idx.get_indexer(keys(s.account_id,s.anchor_month));assert (ix>=0).all()
 grp=d.groupby('SK_ID_PREV',sort=False)
 out['account_observations_to_date']=(grp.cumcount().to_numpy()+1)[ix].astype(float)
 for th in (1,7,30):
  last=d.MONTHS_BALANCE.where(d.SK_DPD.ge(th)).groupby(d.SK_ID_PREV).ffill().to_numpy()[ix]
  out[f'dpd{th}_ever']=np.isfinite(last).astype(float);out[f'dpd{th}_months_since']=s.anchor_month.to_numpy()-last
 ratios={'utilization':('AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'),'payment_min':('AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY'),'payment_balance':('AMT_PAYMENT_TOTAL_CURRENT','AMT_BALANCE'),'drawings_limit':('AMT_DRAWINGS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL'),'receivable_balance':('AMT_TOTAL_RECEIVABLE','AMT_BALANCE'),'atm_share':('AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT'),'pos_share':('AMT_DRAWINGS_POS_CURRENT','AMT_DRAWINGS_CURRENT')}
 for name,(num,den) in ratios.items():
  for pre in ('anchor','lag1'):out[f'ratio_{name}_{pre}']=out[f'{pre}__{num}']/np.maximum(np.abs(out[f'{pre}__{den}']),1)
  out[f'ratio_{name}_delta']=out[f'ratio_{name}_anchor']-out[f'ratio_{name}_lag1']
 for c in ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_TOTAL_RECEIVABLE']:
  out[f'delta1__{c}']=out[f'anchor__{c}']-out[f'lag1__{c}']
  out[f'anchor_minus_w3__{c}']=out[f'anchor__{c}']-out[f'w3_mean__{c}']
  out[f'w3_minus_w12__{c}']=out[f'w3_mean__{c}']-out[f'w12_mean__{c}']
 base=sorted(out);assert len(base)==179,len(base)
 def v(c):return stack[:,:,cols.index(c)]
 B,L,D,P,M,R=[v(c) for c in ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','AMT_TOTAL_RECEIVABLE']]
 short=np.where(M>0,np.maximum(M-P,0)/np.maximum(M,1),0.)
 short[~np.isfinite(M)|~np.isfinite(P)]=np.nan
 monthly={'minimum_shortfall_fraction':short,'payment_balance_fraction':P/np.maximum(np.abs(B),1),'net_draw_balance':(D-P)/np.maximum(np.abs(B),1),'receivable_excess_fraction':(R-v('AMT_RECEIVABLE_PRINCIPAL'))/np.maximum(np.abs(B),1),'utilization':B/np.maximum(np.abs(L),1)}
 for name,a in monthly.items():
  out['flow__'+name+'_current']=a[:,0]
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   for w in (3,6,12):
    out[f'flow__{name}_mean{w}']=np.nanmean(a[:,:w],axis=1)
    out[f'flow__{name}_max{w}']=np.nanmax(a[:,:w],axis=1)
    out[f'flow__{name}_std{w}']=np.nanstd(a[:,:w],axis=1)
   out[f'flow__{name}_change']=a[:,0]-np.nanmean(a[:,1:6],axis=1)
 oldscale=np.maximum(np.abs(L[:,1]),1);oldbal=np.maximum(np.abs(B[:,1]),1)
 out['flow__balance_growth_old_limit']=(B[:,0]-B[:,1])/oldscale
 out['flow__payment_old_balance']=P[:,0]/oldbal
 out['flow__net_draw_old_limit']=(D[:,0]-P[:,0])/oldscale
 out['flow__reconciliation_residual_old_limit']=(B[:,0]-B[:,1]-D[:,0]+P[:,0])/oldscale
 out['flow__limit_change_fraction']=(L[:,0]-L[:,1])/oldscale
 out['flow__over_limit_amount']=np.maximum(B[:,0]-L[:,0],0)
 out['flow__balance_over_old_limit']=B[:,0]/oldscale
 out['flow__minimum_payment_gap']=np.maximum(M[:,0]-P[:,0],0)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  out['flow__shortfall_months_6']=np.nansum(short[:,:6]>0,axis=1).astype(float)
  out['flow__nonpayment_months_6']=np.nansum((P[:,:6]<=0)&(B[:,:6]>0),axis=1).astype(float)
 assert all(not np.isinf(a).any() for a in out.values())
 f=pd.concat([s,pd.DataFrame(out)],axis=1)
 f['row_id']=[f'{c}:{a}:{t}' for c,a,t in s[['customer_id','account_id','anchor_month']].itertuples(index=False)]
 return f,base,sorted(out)

def labels(d,s):
 idx=pd.Index(keys(d.SK_ID_PREV,d.MONTHS_BALANCE));a=d.SK_DPD.to_numpy();res=[]
 for k in (1,2,3):
  ix=idx.get_indexer(keys(s.account_id,s.anchor_month+k));v=np.full(len(s),np.nan);v[ix>=0]=a[ix[ix>=0]];res.append(v)
 a=np.stack(res,axis=1);n=np.isfinite(a).sum(axis=1)
 out=s[['row_id','customer_id','account_id','anchor_month','split']].copy()
 out['followup_months']=n
 for name,threshold in [('y',1),('y30',30)]:
  pos=(a>=threshold).any(axis=1);out[name]=np.where(pos,1.,np.where(n==3,0.,np.nan))
 return out
