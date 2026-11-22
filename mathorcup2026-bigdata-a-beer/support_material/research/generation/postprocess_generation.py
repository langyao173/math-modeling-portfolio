from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd,joblib
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from physical_generator import CEILINGS,feasibility
from run_generation import PROJECT,BASE,atomic_to_csv,writej
from conditional_model import IntervalCopula,DERIVED,TECHNICAL
from sensory_model import predict_bundle
OUT=BASE/'results'

def audit_units(df):
    cal=pd.read_csv(PROJECT/'A_work/audit/calibrated_chemistry_source.csv').set_index('beer').loc[df.beer]
    meta=pd.read_csv(PROJECT/'A_work/audit/chemical_metadata.csv').set_index('field')
    rows=[];diff=[]
    for field,inf in CEILINGS.items():
        x=df[field].to_numpy(float);raw=10**x if inf['transform']=='log10' else x;cc=cal[field].to_numpy(float)
        valid=np.isfinite(raw)&np.isfinite(cc)&(cc>0)
        absdiff=abs(raw-cc);relative=absdiff/np.maximum(abs(cc),1e-100)
        substantive=valid&(absdiff>1e-6)
        rows.append(dict(field=field,unit=meta.loc[field,'unit'],native_transform=meta.loc[field,'transform'],positive_calibration_pairs=int(valid.sum()),maximum_relative_difference=float(relative[valid].max()),substantive_pair_differences=int(substantive.sum()),input_observed_pass=int((np.isfinite(raw)&(raw<=inf['threshold'])).sum()),calibrated_pass=int((np.isfinite(cc)&(cc<=inf['threshold'])).sum()),screening_ceiling=inf['threshold'],attachment_threshold_median=meta.loc[field,'threshold_median'],threshold_origin='Attachment 8 threshold_median' if field!='X.E..2.nonenal' else 'External literature value in existing transfer code; not listed in attachment threshold sheet'))
        for i in np.flatnonzero(substantive):diff.append(dict(beer=int(df.beer.iloc[i]),field=field,input_native=x[i],input_decoded=raw[i],calibrated=cc[i],absolute_difference=absdiff[i],relative_difference=relative[i]))
    pd.DataFrame(rows).to_csv(OUT/'chemical_scale_threshold_audit.csv',index=False);pd.DataFrame(diff).to_csv(OUT/'substantive_calibration_disagreements.csv',index=False)


def bystyle():
    df=pd.read_csv(PROJECT/'data/A/1.csv');scores=pd.read_csv(OUT/'heldout_scores_by_beer.csv');rows=[]
    allstyles=sorted(df.tasting_category_fine.unique())
    for policy in scores.policy.unique():
        table=scores[scores.policy==policy]
        for st in allstyles:
            t=table[table['style']==st];r=t[t.method=='unconditional'];row=dict(policy=policy,style=st,real_validation_n=len(r),real_validation_passing_ceilings=int(r.observed_passes_ceilings.sum()))
            for method in ['unconditional','coordinate_clipping','conditional']:
                v=t[t.method==method];f=v[v.observed_passes_ceilings]
                row[method+'_evaluated_n']=len(v);row[method+'_conditional_evaluated_n']=len(f)
                for metric in ['energy','variogram']:
                    row[method+'_all_'+metric]=v[metric].mean();row[method+'_conditional_'+metric]=f[metric].mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'heldout_quality_by_style.csv',index=False)


def representatives(df):
    features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL];model=IntervalCopula(kind='interval_gaussian',kappa=2,nu=20).fit(df[features],df.tasting_category_fine)
    bundle=joblib.load(PROJECT/'A_work/q1/sensory_ensemble.joblib');basepred=predict_bundle(bundle,df)
    candidates=pd.read_csv(OUT/'attachment_two_conditioned_chemical_profiles.csv');cpred=predict_bundle(bundle,candidates)
    old=pd.read_csv(PROJECT/'A_work/q3/requested_style_representatives.csv');opred=predict_bundle(bundle,old)
    z=model.map_.transform(df[features]);zg=model.map_.transform(candidates[features]);clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(z,df.tasting_category_fine)
    yscale=np.std(bundle['reference_y'],axis=0,ddof=1);stylepred=clf.predict(zg);probs=clf.predict_proba(zg)
    selections=[];profiles=[];sensory=[];chem=[]
    meta=pd.read_csv(PROJECT/'A_work/audit/chemical_metadata.csv').set_index('field')
    fields=['ethanol..v.v.','ethyl_acetate','isopentyl_acetate','phenethyl_alcohol','lactic_acid.mg.L.','acetic_acid.g.L.','bitterness','color.EBC.','DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']
    for st in ['Tripel','Kriek','Lambic']:
        ir=np.flatnonzero(df.tasting_category_fine.to_numpy(str)==st);ig=np.flatnonzero(candidates.tasting_category_fine.to_numpy(str)==st);oi=np.flatnonzero(old.tasting_category_fine.to_numpy(str)==st)[0]
        ref=cdist(z[ir],z[ir])/np.sqrt(len(features));np.fill_diagonal(ref,np.inf);q95=np.quantile(ref.min(axis=1),.95)
        nn=cdist(zg[ig],z[ir]).min(axis=1)/np.sqrt(len(features));target=basepred.iloc[ir].mean().to_numpy(float)
        sens=np.sqrt(np.mean(((cpred.iloc[ig].to_numpy(float)-target)/yscale)[...,:30]**2,axis=1))
        # Fixed engineering selection: first pass the requested label and an
        # empirical same-style support screen; then choose the closest core
        # sensory fingerprint. This is not a generalization evaluation.
        eligible=(stylepred[ig]==st)&(nn<=q95)
        if not eligible.any():eligible=stylepred[ig]==st
        if not eligible.any():eligible=np.ones(len(ig),bool)
        local=np.flatnonzero(eligible)[np.argmin(sens[eligible])];j=ig[local]
        ss=clf.classes_.tolist().index(st)
        row=candidates.iloc[j].copy();row['representative_role']='two_ceiling_conditioned_model_candidate';profiles.append(row)
        selections.append(dict(style=st,selected_id=candidates.iloc[j].synthetic_id,candidate_pool=len(ig),eligible_pool=int(eligible.sum()),predicted_style=stylepred[j],requested_style_probability=float(probs[j,ss]),same_style_nearest_distance=float(nn[local]),same_style_loo_nn_q95=float(q95),core_sensory_distance_to_real_style_prediction_centroid=float(sens[local]),old_DMS_ug_L=float(10**old.iloc[oi].DMS),new_DMS_ug_L=float(10**candidates.iloc[j].DMS),old_diacetyl_native_mg_L=float(old.iloc[oi]['X2.3.butanedione']),new_diacetyl_native_mg_L=float(candidates.iloc[j]['X2.3.butanedione']),style_real_n=len(ir),style_real_two_ceiling_n=int(feasibility(df.iloc[ir][features],features,{k:v for k,v in CEILINGS.items() if k in ['DMS','X2.3.butanedione']})[0].sum())))
        for k,t in enumerate(bundle['targets']):
            v=cpred.iloc[ig,k]
            sensory.append(dict(style=st,target=t,old_representative_prediction=float(opred.iloc[oi,k]),new_representative_prediction=float(cpred.iloc[j,k]),real_style_model_prediction_mean=float(basepred.iloc[ir,k].mean()),conditioned_prediction_q10=float(v.quantile(.1)),conditioned_prediction_median=float(v.median()),conditioned_prediction_q90=float(v.quantile(.9)),reference_response_sd=float(yscale[k]),scale='Frozen six-model predicted standardized sensory value, not new human evaluation'))
        raworig=pd.read_csv(PROJECT/'A_work/q3/revised_generated_chemical_profiles_complete.csv');og=raworig[raworig.tasting_category_fine==st]
        for f in fields:
            sortedreal=np.sort(df[f].dropna().to_numpy(float));ov=float(old.iloc[oi][f]);nv=float(candidates.iloc[j][f]);qs=np.quantile(candidates.iloc[ig][f],[.1,.5,.9]);oq=np.quantile(og[f],[.1,.5,.9]);pct=lambda x:float(np.searchsorted(sortedreal,x,side='right')/len(sortedreal))
            transform=meta.loc[f,'transform'];dec=lambda x:float(10**x if transform=='verified_log10' else x)
            chem.append(dict(style=st,field=f,chemical_name=meta.loc[f,'chemical_name'],unit=meta.loc[f,'unit'],native_transform=transform,old_representative_native=ov,new_representative_native=nv,old_representative_decoded=dec(ov),new_representative_decoded=dec(nv),old_representative_global_percentile=pct(ov),new_representative_global_percentile=pct(nv),original_q10_percentile=pct(oq[0]),original_median_percentile=pct(oq[1]),original_q90_percentile=pct(oq[2]),conditioned_q10_percentile=pct(qs[0]),conditioned_median_percentile=pct(qs[1]),conditioned_q90_percentile=pct(qs[2])))
    pd.DataFrame(profiles).to_csv(OUT/'requested_clean_style_representatives.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'requested_clean_representative_selection.csv',index=False);pd.DataFrame(sensory).to_csv(OUT/'requested_style_sensory_plot_data.csv',index=False);pd.DataFrame(chem).to_csv(OUT/'requested_style_chemistry_plot_data.csv',index=False)


def main():
    global OUT
    parser=argparse.ArgumentParser(description='Audit calibration scales, summarize existing CV, and select three clean-style representatives. Uses MATHORCUP_PROJECT_DIR or ../project/MathorCup_A.')
    parser.add_argument('--out',type=Path,default=BASE/'results',help='Existing run directory within this generation directory')
    parser.add_argument('--skip-cv',action='store_true',help='Do not read or summarize heldout CV files; use after a candidate-only run')
    args=parser.parse_args();OUT=args.out.resolve()
    if not OUT.is_relative_to(BASE):raise ValueError('Output must stay under generation work directory')
    if not OUT.is_dir():raise FileNotFoundError('Run generation first: '+str(OUT))
    if not args.skip_cv and not (OUT/'heldout_scores_by_beer.csv').is_file():raise FileNotFoundError('CV cache missing; run run_generation.py without --skip-cv, or pass --skip-cv here for candidate-only postprocessing.')
    df=pd.read_csv(PROJECT/'data/A/1.csv');audit_units(df)
    if not args.skip_cv:
        bystyle()
        diagnostic_path=OUT/'heldout_distribution_diagnostics.csv'
        if diagnostic_path.is_file():
            diagnostic=pd.read_csv(diagnostic_path)
            diagnostic['distinguishability']=np.maximum(diagnostic.c2st_auc,1-diagnostic.c2st_auc)
            diagnostic.to_csv(diagnostic_path,index=False)
    representatives(df)
    # Every important CSV is reconstructed in memory and atomically persisted,
    # including outputs originally produced by the experiment runner.
    audit=[]
    for path in OUT.glob('*.csv'):
        frame=pd.read_csv(path);frame.to_csv(path,index=False);audit.append(dict(file=path.name,rows=len(frame),bytes=path.stat().st_size))
    writej(audit,OUT/'csv_readback_audit.json')
    print(pd.read_csv(OUT/'requested_clean_representative_selection.csv').to_string(index=False))
if __name__=='__main__':main()
