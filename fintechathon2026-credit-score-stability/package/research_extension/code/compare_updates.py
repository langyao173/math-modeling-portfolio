"""Limited development selection followed by historical-cohort comparison.

No outcome enters an update, an eligibility group, or a case explanation.
"""
from pathlib import Path
import argparse, json, math, hashlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from prepare_inputs import ROOT,CAP,JOINT,dump

GROUPS=['all_transitions','stable_05','repayment_state','flow_structure','new_payment_shortfall','outside_seven','balance_above_reference']
CAPACITIES=[.05,.08,.1,.12,.15,.2]
MODES=['static','bound','upfree','stress_bypass','rank25','rank50','rank75','rank100']
BASES={'reference':'old_reference_FX__static','actual':'old_actual_mean__static'}

def cvar(x):
    a=np.sort(np.asarray(x,float))[::-1];m=.05*len(a);k=int(m)
    return float((a[:k].sum()+(m-k)*a[k])/m) if len(a) else np.nan

def top(p,ids,cap):return np.lexsort((np.asarray(ids,dtype=str),-np.asarray(p)))[:math.ceil(cap*len(p))]

def selected(p,meta,cap,development):
    if not development:return top(p,meta.row_id,cap)
    folds=meta.fold.to_numpy();sizes=np.array([(folds==f).sum() for f in range(3)])
    ideal=cap*sizes;quotas=np.floor(ideal).astype(int);order=np.lexsort((np.arange(3),-(ideal-quotas)))
    quotas[order[:int(np.ceil(cap*len(meta))-quotas.sum())]]+=1
    out=[]
    for f,k in enumerate(quotas):
        ix=np.flatnonzero(folds==f);rank=np.lexsort((meta.row_id.to_numpy(str)[ix],-p[ix]));out.extend(ix[rank[:k]])
    return np.asarray(out,int)

def update(p,meta,edges,mode):
    q=np.asarray(p,float).copy()
    if mode=='static':return q
    for phase in [0,1]:
        for fold in sorted(meta.loc[meta.fold.ge(0),'fold'].unique()):
            e=edges[edges.phase_after.eq(phase)&edges.fold.eq(fold)]
            gate=e.seven_05.to_numpy();i=e.row_before.to_numpy(int);j=e.row_after.to_numpy(int)
            if mode=='stress_bypass':gate &= ~e.new_payment_shortfall.to_numpy()
            if mode.startswith('rank'):
                w=int(mode[4:])/100
                rows=np.flatnonzero(meta.phase.eq(phase)&meta.fold.eq(fold))
                target=q[rows].copy();pos=pd.Series(np.arange(len(rows)),index=rows)
                target[pos.loc[j[gate]].to_numpy()]=(1-w)*p[j[gate]]+w*q[i[gate]]
                iso=IsotonicRegression(y_min=1e-10,y_max=1-1e-10,out_of_bounds='clip').fit_transform(p[rows],target)
                q[rows]=(1-1e-6)*iso+1e-6*p[rows]
            elif mode=='upfree':q[j[gate]]=np.maximum(p[j[gate]],q[i[gate]]-.01)
            else:q[j[gate]]=np.clip(p[j[gate]],q[i[gate]]-.01,q[i[gate]]+.01)
    return q

def summarize(scores,meta,edges,y,development):
    risk=[];tail=[];caprows=[];membership=[]
    for phase in [0,1]:
        mask=meta.phase.eq(phase)&meta.fold.ge(0);m=meta.loc[mask].copy();yy=y[mask];known=np.isfinite(yy)
        for name,prob in scores.items():
            pp=prob[mask];r=dict(candidate=name,phase=phase,n=len(m),known=int(known.sum()),events=int(np.nansum(yy)),auc=roc_auc_score(yy[known],pp[known]),ap=average_precision_score(yy[known],pp[known]),brier=np.mean((yy[known]-pp[known])**2),logloss=log_loss(yy[known],pp[known],labels=[0,1]))
            for c in CAPACITIES:
                idx=selected(pp,m,c,development);hits=int(np.nansum(yy[idx]));caprows.append(dict(candidate=name,phase=phase,capacity=c,n=len(m),slots=len(idx),hits=hits,unknown=int(np.isnan(yy[idx]).sum()),recall=hits/np.nansum(yy)))
                if c==.1:
                    r['hits10']=hits
                    membership.extend(dict(candidate=name,phase=phase,customer_id=int(m.iloc[z].customer_id),row_id=m.iloc[z].row_id,y=yy[z],probability=pp[z]) for z in idx)
            risk.append(r)
            for group in GROUPS:
                e=edges[edges.phase_after.eq(phase)&edges.fold.ge(0)&edges[group]];ii=e.row_before.to_numpy(int);jj=e.row_after.to_numpy(int);a=100*abs(prob[jj]-prob[ii])
                tail.append(dict(candidate=name,phase=phase,group=group,n=len(e),seven_n=int(e.seven_05.sum()),seven_overlap=float(e.seven_05.mean()) if len(e) else np.nan,mean_pp=float(a.mean()) if len(a) else np.nan,p95_pp=float(np.quantile(a,.95)) if len(a) else np.nan,cvar95_pp=cvar(a),increase_percent=float(np.mean(prob[jj]>prob[ii])*100) if len(e) else np.nan,events=int(np.nansum(y[jj])),known=int(np.isfinite(y[jj]).sum())))
    return pd.DataFrame(risk),pd.DataFrame(tail),pd.DataFrame(caprows),pd.DataFrame(membership)

def select(r,t,c):
    base=r[r.candidate.eq('actual_static')].set_index('phase');b=base[['auc','ap','brier','logloss']].mean();rows=[]
    groups=['stable_05','repayment_state','flow_structure']
    norm=t[t.candidate.eq('reference_static')&t.group.isin(groups)].set_index(['phase','group']).cvar95_pp
    current=t[t.candidate.eq('reference_bound')&t.group.eq('stable_05')].cvar95_pp.max()
    for name,g in r.groupby('candidate'):
        a=g.set_index('phase');v=a[['auc','ap','brier','logloss']].mean();s=t[t.candidate.eq(name)&t.group.isin(groups)].set_index(['phase','group']).cvar95_pp
        checks={'auc':v.auc>=b.auc-.005,'ap':v.ap>=.98*b.ap,'brier':v.brier<=1.02*b.brier,'logloss':v.logloss<=1.02*b.logloss,'coverage':a.hits10.sum()>=base.hits10.sum(),'phase':bool((a.hits10>=base.hits10-2).all()),'capacity':c[c.candidate.eq(name)].recall.mean()>=.98*c[c.candidate.eq('actual_static')].recall.mean(),'tail':t[t.candidate.eq(name)&t.group.eq('stable_05')].cvar95_pp.max()<=1.25*current}
        rows.append(dict(candidate=name,**v.to_dict(),hits10=int(a.hits10.sum()),worst_relative_tail=float((s/norm).max()),worst_original_tail=float(s.xs('stable_05',level='group').max()),eligible=all(checks.values()),**{'guard_'+k:v for k,v in checks.items()}))
    table=pd.DataFrame(rows);eligible=table[table.eligible&table.candidate.str.startswith('reference_')]
    ordered=eligible.sort_values(['worst_relative_tail','ap','hits10','candidate'],ascending=[True,False,False,True])
    return table,{'selected':ordered.iloc[0].candidate if len(ordered) else 'reference_bound','new_candidate_passed':bool(len(ordered) and ordered.iloc[0].candidate!='reference_bound'),'eligible_reference_count':len(ordered),'selection_inputs':'customer-out-of-fold development predictions only'}

def feasibility(p,meta,edges):
    rows=[];witness=[]
    # Intervals around prior static p isolate the first-transition conflict.
    # Inactive cases are held at their current p. Nondecreasing feasibility is
    # a necessary condition for strict ranking preservation.
    for phase in [0,1]:
        for fold in sorted(meta.loc[meta.fold.ge(0),'fold'].unique()):
            idx=np.flatnonzero(meta.phase.eq(phase)&meta.fold.eq(fold));m=meta.iloc[idx]
            e=edges[edges.phase_after.eq(phase)&edges.fold.eq(fold)&edges.seven_05]
            lo=p[idx].copy();hi=p[idx].copy();pos=pd.Series(np.arange(len(idx)),index=idx)
            for a,b in zip(e.row_before,e.row_after):
                z=pos.loc[b];lo[z]=max(0,p[a]-.01);hi[z]=min(1,p[a]+.01)
            order=np.lexsort((m.row_id.to_numpy(str),p[idx]));running=-np.inf;arg=-1;bad=0
            for z in order:
                if lo[z]>running:running=lo[z];arg=z
                if running>hi[z]+1e-12:
                    bad+=1
                    witness.append(dict(phase=phase,fold=int(fold),lower_rank_row=m.iloc[arg].row_id,higher_rank_row=m.iloc[z].row_id,required_lower=float(running),allowed_upper=float(hi[z]),gap_pp=float(100*(running-hi[z]))))
            rows.append(dict(phase=phase,fold=int(fold),n=len(idx),incompatible_positions=bad,feasible=bad==0))
    return pd.DataFrame(rows),pd.DataFrame(witness)

def main(stage):
    development=stage=='development';short='development' if development else 'historical';out=ROOT/'results'/short;out.mkdir(exist_ok=True)
    meta=pd.read_parquet(ROOT/'results'/f'{short}_meta.parquet');edges=pd.read_parquet(ROOT/'results'/f'{short}_edges.parquet')
    group='event_development' if development else 'event_evaluation'
    yp=ROOT/'results'/f'{short}_outcomes.parquet'
    if not yp.exists():yp=CAP/'data'/group/'outcomes.parquet' if development else CAP/'data/evaluation_outcomes/event_evaluation/outcomes.parquet'
    y=pd.read_parquet(yp).y.to_numpy(float)
    if not development:assert (ROOT/'results/development/selection.json').exists()
    path=out/'frozen_inputs.parquet'
    if not path.exists():path=JOINT/('results/development/oof_predictions.parquet' if development else 'results/final/predictions_frozen.parquet')
    frozen=pd.read_parquet(path);assert np.array_equal(meta.sample_id,frozen.sample_id)
    scores={}
    for base,col in BASES.items():
        p=frozen[col].to_numpy()
        for mode in MODES:scores[base+'_'+mode]=update(p,meta,edges,mode)
    if not development:
        for col in ['incumbent_R','previous_tail','previous_FX','previous_E']:scores[col]=frozen[col].to_numpy()
    np.testing.assert_allclose(scores['reference_bound'][meta.fold.ge(0)],frozen.loc[meta.fold.ge(0),'old_reference_FX__bound10'],rtol=1e-12,atol=1e-14)
    r,t,c,m=summarize(scores,meta,edges,y,development)
    for name,frame in [('risk',r),('tail',t),('capacity',c),('memberships',m)]:frame.to_csv(out/f'{name}.csv',index=False)
    pd.concat([meta[['sample_id','row_id','customer_id','phase','fold']],pd.DataFrame(scores)],axis=1).to_parquet(out/'predictions.parquet',index=False)
    if development:
        table,choice=select(r,t,c);table.to_csv(out/'selection_table.csv',index=False)
        choice['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();dump(out/'selection.json',choice)
        print(json.dumps(choice,ensure_ascii=False),flush=True);print(table.to_string(index=False),flush=True)
    else:
        f,w=feasibility(scores['reference_static'],meta,edges);f.to_csv(out/'rank_bound_feasibility.csv',index=False);w.to_csv(out/'rank_bound_witnesses.csv',index=False)
        print(r.to_string(index=False),flush=True)
    # Each deployment batch preserves order. Development has three independently
    # transformed folds, so pooled OOF AUC/AP need not be invariant across folds.
    checks=[]
    for base in BASES:
        for mode in ['rank25','rank50','rank75','rank100']:
            for phase in [0,1]:
                for fold in sorted(meta.loc[meta.fold.ge(0),'fold'].unique()):
                    k=meta.phase.eq(phase)&meta.fold.eq(fold)&np.isfinite(y)
                    pa=scores[base+'_static'][k];pb=scores[base+'_'+mode][k]
                    for metric,fn in [('auc',roc_auc_score),('ap',average_precision_score)]:
                        a=fn(y[k],pa);b=fn(y[k],pb)
                        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
                        checks.append(dict(candidate=base+'_'+mode,phase=phase,fold=int(fold),metric=metric,difference=float(b-a)))
            aa=c[c.candidate.eq(base+'_'+mode)].sort_values(['phase','capacity']);bb=c[c.candidate.eq(base+'_static')].sort_values(['phase','capacity'])
            assert np.array_equal(aa.hits.to_numpy(),bb.hits.to_numpy())
            for phase in [0,1]:
                kk=meta.phase.eq(phase)&meta.fold.ge(0);mm=meta.loc[kk]
                for capacity in CAPACITIES:
                    a=selected(scores[base+'_static'][kk],mm,capacity,development)
                    b=selected(scores[base+'_'+mode][kk],mm,capacity,development)
                    assert np.array_equal(a,b), (base,mode,phase,capacity,'list changed')
    pd.DataFrame(checks).to_csv(out/'rank_preservation_checks.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','historical']);main(parser.parse_args().stage)
