from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time,warnings
from dataclasses import asdict
import numpy as np,pandas as pd,joblib
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from physical_generator import CEILINGS,feasibility,CeilingConditionedIntervalGaussian,wilson

# Complete buffering and atomic writeback avoid partial files in snapshotting
# worker filesystems. Every CSV is immediately reopened to verify row count.
_original_to_csv=pd.DataFrame.to_csv
def atomic_to_csv(self,path_or_buf=None,*args,**kwargs):
    if isinstance(path_or_buf,(str,Path)):
        path=Path(path_or_buf);text=_original_to_csv(self,None,*args,**kwargs)
        temp=path.with_name(path.name+'.atomic.tmp')
        payload=text.encode('utf-8')
        with temp.open('wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        if path.read_bytes()!=payload:raise IOError('Atomic CSV bytes differ: '+str(path))
        parsed=pd.read_csv(path)
        if len(parsed)!=len(self):raise IOError('Atomic CSV row count differs: '+str(path))
        return None
    return _original_to_csv(self,path_or_buf,*args,**kwargs)
pd.DataFrame.to_csv=atomic_to_csv

BASE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('MATHORCUP_PROJECT_DIR',str(BASE.parent/'project/MathorCup_A'))).expanduser().resolve()
sys.path[:0]=[str(PROJECT/'A_work/q3'),str(PROJECT/'A_work/q1')]
from conditional_model import IntervalCopula,DERIVED,TECHNICAL,energy_scores,variogram_scores
from sensory_model import predict_bundle
from transfer_profiles import enrich
SEED=20261025
POLICIES={'attachment_two':{k:v for k,v in CEILINGS.items() if k in ['DMS','X2.3.butanedione']},'strict_four':CEILINGS}

def writej(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else x))

def cap_clip(x,features,ceilings):
    x=x.copy()
    for f,inf in ceilings.items():
        j=features.index(f);t=np.log10(inf['threshold']) if inf['transform']=='log10' else inf['threshold'];x[:,j]=np.minimum(x[:,j],t)
    return x

def complete(frame,df,features,pack):
    return enrich(frame,df,features,pack['model'].map_,pack['derived_predictors'])

def describe_candidates(df,frames,features,model,ceilings,bundle,policy,out):
    z=model.map_.transform(df[features]);styles=df.tasting_category_fine.to_numpy(str)
    clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=3000).fit(z,styles)
    yscale=np.std(bundle['reference_y'],axis=0,ddof=1)
    keep=[j for j,t in enumerate(bundle['targets']) if t not in ['dms','diacetyl','X4vg','t2n']]
    core=list(range(30));predictions={};rows=[];profiles=[]
    for method,frame in frames.items():
        if not len(frame):continue
        predictions[method]=predict_bundle(bundle,frame).to_numpy(float)
        zz=model.map_.transform(frame[features]);label=clf.predict(zz)
        nn=cdist(zz,z)/np.sqrt(len(features));glob=nn.min(axis=1)
        valid,_=feasibility(frame[features],features,ceilings)
        frame[[c for c in ['synthetic_id','tasting_category_fine'] if c in frame]].assign(**{t:predictions[method][:,j] for j,t in enumerate(bundle['targets'])}).to_csv(out/f'{policy}_{method}_sensory_predictions.csv',index=False)
        for style in model.styles_:
            ix=np.flatnonzero(frame.tasting_category_fine.to_numpy(str)==style);ir=np.flatnonzero(styles==style)
            if not len(ix):continue
            dd=cdist(z[ir],z[ir])/np.sqrt(len(features));np.fill_diagonal(dd,np.inf)
            ref=np.median(dd.min(axis=1));q95=np.quantile(dd.min(axis=1),.95)
            same=nn[np.ix_(ix,ir)].min(axis=1)
            baseline=frames['original_unconditional'];bi=np.flatnonzero(baseline.tasting_category_fine.to_numpy(str)==style)
            center=predictions['original_unconditional'][bi].mean(axis=0)
            cur=predictions[method][ix].mean(axis=0);shift=(cur-center)/yscale
            rows.append(dict(policy=policy,method=method,style=style,n=len(ix),ceiling_pass_rate=float(valid[ix].mean()),style_classifier_retention=float(np.mean(label[ix]==style)),exact_copy_rate=float(np.mean(glob[ix]<1e-8)),global_nearest_real_median=float(np.median(glob[ix])),same_style_nearest_real_median=float(np.median(same)),same_style_nn_ratio_to_real_loo=float(np.median(same)/ref),fraction_above_same_style_loo_q95=float(np.mean(same>q95)),sensory_centroid_shift_all50=float(np.sqrt(np.mean(shift**2))),sensory_centroid_shift_core30=float(np.sqrt(np.mean(shift[core]**2))),sensory_centroid_shift_nondefect46=float(np.sqrt(np.mean(shift[keep]**2))),predicted_overall_mean=float(cur[-1]),predicted_stale_hops_mean=float(cur[bundle['targets'].index('stale_hops')]),empirical_real_style_n=len(ir)))
            for j in ix:
                profiles.append(dict(policy=policy,method=method,synthetic_id=frame.iloc[j].synthetic_id,style=style,passes_ceilings=bool(valid[j]),predicted_style=label[j],nearest_training_distance=float(glob[j])))
    pd.DataFrame(rows).to_csv(out/f'{policy}_candidate_summary.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/f'{policy}_candidate_diagnostics.csv',index=False)
    return pd.DataFrame(rows)

def cv_evaluation(df,features,out):
    X=df[features].to_numpy(float);s=df.tasting_category_fine.to_numpy(str);rows=[];foldrows=[];details=[];support=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261023+300).split(X,s)):
        model=IntervalCopula(kind='interval_gaussian',kappa=2,nu=20).fit(X[tr],s[tr]);zv=model.map_.transform(X[va]);rng=np.random.default_rng(SEED+500+fold)
        pairs=rng.integers(0,len(features),(1000,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
        for policy,ceilings in POLICIES.items():
            feasible_va,_=feasibility(X[va],features,ceilings)
            try:conditional=CeilingConditionedIntervalGaussian(model,features,ceilings);err=None
            except ValueError as ex:conditional=None;err=str(ex)
            # Each style's validation sample is scored once, with fold-local transforms.
            zg_pool={k:[] for k in ['unconditional','coordinate_clipping','conditional']};truth_pool=[]
            for st in np.unique(s[va]):
                ids=np.flatnonzero(s[va]==st);truth=zv[ids];is_feasible=feasible_va[ids]
                g,gz=model.sample(st,512,rng)
                clipped=cap_clip(g,features,ceilings);zc=model.map_.transform(clipped)
                samples={'unconditional':gz,'coordinate_clipping':zc}
                if conditional is not None:
                    cg,cz,rec=conditional.sample(st,512,rng,max_proposals=5000000)
                    if rec.complete:samples['conditional']=cz
                    support.append(dict(fold=fold,policy=policy,style=st,validation_n=len(ids),validation_feasible_n=int(is_feasible.sum()),conditioned_available=bool(rec.complete),sampled=rec.accepted,proposed=rec.proposed,error='' if rec.complete else 'proposal budget exhausted'))
                else:support.append(dict(fold=fold,policy=policy,style=st,validation_n=len(ids),validation_feasible_n=int(is_feasible.sum()),conditioned_available=False,sampled=0,proposed=0,error=err))
                for method,draws in samples.items():
                    es=energy_scores(truth,draws);vs=variogram_scores(truth,draws,pairs)
                    for j,ii in enumerate(ids):rows.append(dict(fold=fold,policy=policy,method=method,beer=int(df.iloc[va[ii]].beer),style=st,observed_passes_ceilings=bool(is_feasible[j]),energy=float(es[j]),variogram=float(vs[j])))
                    if is_feasible.any():zg_pool[method].append(draws[:int(is_feasible.sum())])
                if is_feasible.any():truth_pool.append(truth[is_feasible])
            if truth_pool:
                real=np.vstack(truth_pool)
                for method,blocks in zg_pool.items():
                    if not blocks:continue
                    gen=np.vstack(blocks)
                    if len(gen)!=len(real):continue
                    nnr=cdist(real,model.z_).min(axis=1)/np.sqrt(len(features));nng=cdist(gen,model.z_).min(axis=1)/np.sqrt(len(features))
                    auc=np.nan
                    if len(real)>=8:
                        A=np.vstack([real,gen]);labels=np.r_[np.zeros(len(real)),np.ones(len(gen))]
                        cf=StratifiedKFold(min(4,len(real)),shuffle=True,random_state=SEED+fold)
                        disc=RandomForestClassifier(n_estimators=120,max_depth=4,min_samples_leaf=3,n_jobs=1,random_state=SEED+fold)
                        pr=cross_val_predict(disc,A,labels,method='predict_proba',cv=cf)[:,1];auc=float(roc_auc_score(labels,pr))
                    foldrows.append(dict(fold=fold,policy=policy,method=method,conditional_real_n=len(real),c2st_auc=auc,nn_ratio=float(np.median(nng)/np.median(nnr)),exact_copy_rate=float(np.mean(nng<1e-8))))
        pd.DataFrame(rows).to_csv(out/'heldout_scores_by_beer.csv',index=False);pd.DataFrame(foldrows).to_csv(out/'heldout_distribution_diagnostics.csv',index=False);pd.DataFrame(support).to_csv(out/'heldout_conditional_support.csv',index=False)
        print('CV fold complete',fold,flush=True)
    scores=pd.DataFrame(rows);summary=[]
    for (policy,method),v in scores.groupby(['policy','method']):
        for target,mask in [('all_validation',np.ones(len(v),bool)),('ceiling_passing_validation',v.observed_passes_ceilings.to_numpy(bool))]:
            vv=v.loc[mask]
            summary.append(dict(policy=policy,method=method,evaluation_population=target,validation_beers=len(vv),unique_beers=vv.beer.nunique(),energy=float(vv.energy.mean()),variogram=float(vv.variogram.mean())))
    tab=pd.DataFrame(summary);tab.to_csv(out/'heldout_quality_summary.csv',index=False)
    # Paired conditional comparisons use only the same feasible validation beers.
    intervals=[]
    for policy in POLICIES:
        t=scores[(scores.policy==policy)&scores.observed_passes_ceilings]
        pivot=t.pivot(index='beer',columns='method',values='energy')
        if 'conditional' not in pivot:continue
        for ref in ['unconditional','coordinate_clipping']:
            v=pivot[['conditional',ref]].dropna();delta=v['conditional'].to_numpy()-v[ref].to_numpy()
            if not len(delta):continue
            bs=delta[np.random.default_rng(SEED).integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
            intervals.append(dict(policy=policy,reference=ref,comparison='conditional minus reference; negative favors conditional',beers=len(delta),energy_delta=float(delta.mean()),ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975))))
    pd.DataFrame(intervals).to_csv(out/'heldout_paired_energy_intervals.csv',index=False)
    return tab

def main():
    ap=argparse.ArgumentParser(description='Generate and assess fixed two-ceiling candidates; optionally run training-fold-only CV. Project root uses MATHORCUP_PROJECT_DIR or ../project/MathorCup_A.');ap.add_argument('--out',type=Path,default=BASE/'results',help='Output directory within this generation directory');ap.add_argument('--skip-cv',action='store_true',help='Generate candidates and descriptors only; do not run or refresh heldout CV files');args=ap.parse_args();out=args.out.resolve()
    if not out.is_relative_to(BASE):raise ValueError('Output must stay under generation work directory')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    df=pd.read_csv(PROJECT/'data/A/1.csv');features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL];X=df[features].to_numpy(float);s=df.tasting_category_fine.to_numpy(str)
    original=pd.read_csv(PROJECT/'A_work/q3/revised_generated_chemical_profiles_complete.csv');model=IntervalCopula(kind='interval_gaussian',kappa=2,nu=20).fit(X,s)
    pack=joblib.load(PROJECT/'A_work/q3/transfer_and_augmentation_models.joblib');bundle=joblib.load(PROJECT/'A_work/q1/sensory_ensemble.joblib')
    writej(dict(seed=SEED,model=dict(kind='interval_gaussian',kappa=2,nu=20),policies=POLICIES,primary_policy='attachment_two',strict_four='Sensitivity scenario; the nonenal threshold is external and only one measured beer passes. Never report this as a universal style feasibility standard.',sample_law='P_style(X | specified ceilings); exact four/two-dimensional marginal rejection plus conditional Gaussian for other coordinates',validation='Post-diagnostic internal three-fold evaluation, no independent new beer data. Fit maps and covariances on training folds only. Conditional-law quality must be evaluated against feasible held-out real beers, not mixed with unconditional-law scores.',sensory='Frozen six-model full-data predictions are descriptive, not independent tasting validation.',original_generated_count=len(original),real_n=len(df)),out/'protocol.json')
    passrows=[];sampling=[];allsummary=[]
    for pi,(policy,ceilings) in enumerate(POLICIES.items()):
        rp,rc=feasibility(X,features,ceilings);gp,gc=feasibility(original[features],features,ceilings)
        for st in model.styles_:
            ir=np.flatnonzero(s==st);ig=np.flatnonzero(original.tasting_category_fine.to_numpy(str)==st)
            row=dict(policy=policy,style=st,real_n=len(ir),real_pass=int(rp[ir].sum()),real_pass_rate=float(rp[ir].mean()),original_n=len(ig),original_pass=int(gp[ig].sum()),original_pass_rate=float(gp[ig].mean()),filtered_retained=int(gp[ig].sum()),filtered_conditional_pass_rate=1. if gp[ig].sum() else np.nan)
            for j,f in enumerate(ceilings):row[f+'_real_pass']=int(rc[ir,j].sum());row[f+'_original_pass']=int(gc[ig,j].sum())
            passrows.append(row)
        conditioned=CeilingConditionedIntervalGaussian(model,features,ceilings);writej(conditioned.bounds,out/f'{policy}_latent_caps.json')
        blocks=[]
        for si,st in enumerate(model.styles_):
            rng=np.random.default_rng(SEED+pi*100000+si)
            est=conditioned.estimate_acceptance(st,np.random.default_rng(SEED+900000+pi*1000+si),n=100000)
            x,z,rec=conditioned.sample(st,100,rng,max_proposals=5000000)
            f=pd.DataFrame(x,columns=features);f.insert(0,'tasting_category_fine',st);f.insert(0,'synthetic_id',[f'{policy}_{st.replace("/","_")}_{j+1:03d}' for j in range(len(f))]);blocks.append(f)
            sampling.append(dict(policy=policy,style=st,**asdict(rec),fixed_mc_estimate=est['estimate'],fixed_mc_ci_low=est['ci_low'],fixed_mc_ci_high=est['ci_high'],fixed_mc_trials=est['trials'],fixed_mc_successes=est['successes']))
            print('Generated',policy,st,rec.accepted,'proposals',rec.proposed,'fixed acceptance',est['estimate'],flush=True)
        candidates=complete(pd.concat(blocks,ignore_index=True),df,features,pack);candidates.to_csv(out/f'{policy}_conditioned_chemical_profiles.csv',index=False)
        filtered=original.loc[gp].copy();clipped=original.copy();clipped[features]=cap_clip(original[features].to_numpy(float),features,ceilings);clipped=complete(clipped,df,features,pack)
        filtered.to_csv(out/f'{policy}_filtered_original_profiles.csv',index=False)
        frames={'original_unconditional':original,'same_pool_filtered':filtered,'coordinate_clipping':clipped,'conditioned':candidates}
        allsummary.append(describe_candidates(df,frames,features,model,ceilings,bundle,policy,out))
        pd.DataFrame(passrows).to_csv(out/'ceiling_pass_rates_by_style.csv',index=False);pd.DataFrame(sampling).to_csv(out/'conditional_sampling_efficiency.csv',index=False)
    pd.concat(allsummary,ignore_index=True).to_csv(out/'candidate_summary_all.csv',index=False)
    q=None if args.skip_cv else cv_evaluation(df,features,out)
    writej(dict(complete=True,elapsed_seconds=time.perf_counter()-start,n_original=len(original),requested_conditioned_per_policy=2200,actual_conditioned_counts={p:len(pd.read_csv(out/f'{p}_conditioned_chemical_profiles.csv')) for p in POLICIES},cv_complete=not args.skip_cv),out/'run_summary.json')
    print('ALL COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
