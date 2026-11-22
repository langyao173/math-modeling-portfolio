"""Task-focused diagnostics from saved, executed predictions and bootstrap draws.

No model is selected from these evaluation results. Relative grades refer to
training-fold seven-quantile bins, not recovered raw panel scores.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, math, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, r2_score

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
DATA=Path(os.environ.get('MATHORCUP_DATA_DIR',WORK.parent/'data/A'))
sys.path.insert(0,str(WORK/'q2'))
from run_q2 import greedy_modularity

def dump(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def grading():
    out=HERE/'grading';out.mkdir(parents=True,exist_ok=True)
    med=np.load(WORK/'audit/model_validation/strict_missingness_oof.npz')
    knn=np.load(WORK/'audit/model_validation/strict_knn_missingness_oof.npz')
    X=pd.read_csv(DATA/'1.csv').sort_values('beer').reset_index(drop=True)
    yd=pd.read_csv(DATA/'4.csv').set_index('beer').loc[X.beer]
    targets=yd.columns[2:].tolist();Y=yd[targets].to_numpy(float)
    np.testing.assert_allclose(Y,med['Y'],atol=1e-12)
    np.testing.assert_allclose(Y,knn['Y'],atol=1e-12)
    assert np.array_equal(med['beer'],X.beer)
    assert np.array_equal(med['fold'],knn['fold'])
    folds=med['fold'];cut=np.zeros((len(Y),6,len(targets)));base=np.zeros_like(Y)
    for f in np.unique(folds):
        tr=folds!=f;te=~tr
        cut[te]=np.quantile(Y[tr],np.arange(1,7)/7,axis=0)
        base[te]=Y[tr].mean(axis=0)
    def grade(pred):return (pred[:,None,:]>=cut).sum(axis=1)
    true=grade(Y);groups={'全部50维':np.ones(50,bool),'核心30维':np.array([t.startswith(('A_','F_')) for t in targets]),'香气15维':np.array([t.startswith('A_') for t in targets]),'风味15维':np.array([t.startswith('F_') for t in targets])}
    rows=[];dim=[];conf=[]
    arrays={('训练均值','original'):base}
    for name,arr in [('中位数插补',med),('KNN插补',knn)]:
        for s in ['original','derived_only_missing','MCAR_10_strict','MCAR_30_strict','MCAR_50_strict']:
            arrays[name,s]=arr[s]
    for (method,scenario),pred in arrays.items():
        pg=grade(pred);err=np.abs(pg-true)
        for group,mask in groups.items():
            # Each target has its own training-only bins; all are 0--6 relative grades.
            kappa=[cohen_kappa_score(true[:,j],pg[:,j],labels=np.arange(7),weights='quadratic') for j in np.flatnonzero(mask)]
            rows.append(dict(method=method,scenario=scenario,target_group=group,n_beers=len(Y),n_targets=int(mask.sum()),macro_r2=float(r2_score(Y[:,mask],pred[:,mask],multioutput='uniform_average')),grade_MAE=float(err[:,mask].mean()),exact_fraction=float((err[:,mask]==0).mean()),within_one_fraction=float((err[:,mask]<=1).mean()),macro_quadratic_kappa=float(np.nanmean(kappa))))
        for j,t in enumerate(targets):
            dim.append(dict(method=method,scenario=scenario,target=t,grade_MAE=float(err[:,j].mean()),within_one_fraction=float((err[:,j]<=1).mean()),r2=float(r2_score(Y[:,j],pred[:,j]))))
        if scenario in ['original','MCAR_50_strict']:
            mask=groups['核心30维']
            for observed in range(7):
                for predicted in range(7):
                    conf.append(dict(method=method,scenario=scenario,true_relative_grade=observed,predicted_relative_grade=predicted,count=int(((true[:,mask]==observed)&(pg[:,mask]==predicted)).sum())))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(dim).to_csv(out/'dimension_metrics.csv',index=False)
    pd.DataFrame(conf).to_csv(out/'core30_confusion.csv',index=False)
    pd.concat([X[['beer','tasting_category_fine']],pd.DataFrame(grade(knn['original']),columns=targets)],axis=1).to_csv(out/'knn_oof_relative_grades_50.csv',index=False)
    # Fixed source also used in the transfer worked example; no best-case selection.
    i=int(np.flatnonzero(X.beer.to_numpy()==90)[0]);case=[]
    pg=grade(knn['original'])
    for t in ['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all']:
        j=targets.index(t)
        case.append(dict(beer=90,fold=int(folds[i]),target=t,predicted_continuous=float(knn['original'][i,j]),observed_continuous=float(Y[i,j]),predicted_relative_grade=int(pg[i,j]),observed_relative_grade=int(true[i,j])))
    pd.DataFrame(case).to_csv(out/'beer90_case.csv',index=False)
    summary={'n_beers':250,'n_targets':50,'n_core_targets':30,'cutpoints':'Six seven-quantile cutpoints fitted to the 200 outer-training rows separately for each target and fold. Validation labels are binned using the same training cutpoints.','score_meaning':'Relative intensity grades on the provided standardized aggregate labels. Not original 0-6 panel scores.','evaluation_scope':'Reanalysis of saved outer-fold predictions; no retraining and no new independent test. KNN specs and weights fixed from original median-imputation model.','baseline':'Outer-training target mean, binned with the same fold cutpoints.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'audit/model_validation/strict_missingness_oof.npz',WORK/'audit/model_validation/strict_knn_missingness_oof.npz']}}
    dump(summary,out/'summary.json')
    print(pd.DataFrame(rows).query("target_group=='核心30维'").to_string(index=False),flush=True)

def modules():
    out=HERE/'module_stability';out.mkdir(parents=True,exist_ok=True)
    d=np.load(WORK/'q2/bootstrap_coefficients.npz');B=d['coefficients'];features=d['features'].tolist();targets=d['targets'].tolist()
    node=pd.read_csv(WORK/'q2/graph_nodes.csv').set_index('field').loc[features]
    eligible=np.flatnonzero(node.single_compound_candidate.to_numpy(bool))
    # Use exactly the stored draws and stable top-3 rule, preserving co-selection.
    top=eligible[np.argsort(-np.abs(B[:,eligible,:]),axis=1,kind='stable')[:,:3,:]]
    freq=np.stack([(top==j).any(axis=1).mean(axis=0) for j in range(len(features))])
    evidence=pd.read_csv(WORK/'q2/all_factor_evidence.csv')
    old=evidence.pivot(index='field',columns='target',values='bootstrap_top3_frequency').loc[features,targets].to_numpy()
    err=float(np.max(abs(freq-old)))
    assert err<=1/400+1e-12,'Stored float32 draws differ by more than one rank tie'
    R=pd.read_csv(DATA/'2.csv',index_col=0).to_numpy(float)
    allrows=[];bestrows=[];members=[];targets_rows=[]
    for th in [.5,.6,.7,.8]:
        W=np.where(abs(R)>=th,abs(R),0.);np.fill_diagonal(W,0.)
        labs=greedy_modularity(W)
        if th==.6:assert np.array_equal(labs,node.module.to_numpy())
        topmodules=labs[top]
        for m in sorted(set(labs[eligible])):
            ix=eligible[labs[eligible]==m];ms=len(ix)
            null=1-(math.comb(len(eligible)-ms,3)/math.comb(len(eligible),3) if len(eligible)-ms>=3 else 0)
            mf=(topmodules==m).any(axis=1).mean(axis=0)
            maxf=freq[ix].max(axis=0)
            if th==.6:
                for j in ix:
                    members.append(dict(module=int(m),field=features[j],name=node.iloc[j]['name'],origin=node.iloc[j].origin,module_size_eligible=ms))
            for j,t in enumerate(targets):
                bestix=ix[np.argmax(freq[ix,j])]
                allrows.append(dict(threshold=th,target=t,module=int(m),eligible_module_size=ms,module_union_frequency=float(mf[j]),largest_member_frequency=float(maxf[j]),substitution_gain=float(mf[j]-maxf[j]),uniform_three_draw_reference=float(null),chance_adjusted_frequency=float((mf[j]-null)/(1-null)) if null<1 else None,leading_field=features[bestix],leading_name=node.iloc[bestix]['name'],dominant_origin=str(node.iloc[ix].origin.value_counts().index[0])))
    table=pd.DataFrame(allrows);table.to_csv(out/'module_target_stability.csv',index=False)
    pd.DataFrame(members).to_csv(out/'module_members.csv',index=False)
    # A union event is easier than a single-compound event; expose size and the
    # uniform-draw reference, and do not call the difference a statistical test.
    for (th,t),part in table.groupby(['threshold','target'],sort=False):
        part=part.sort_values(['module_union_frequency','chance_adjusted_frequency','eligible_module_size','module'],ascending=[False,False,True,True],kind='stable')
        r=part.iloc[0].to_dict();r['any_individual_at_least_0_8']=bool((freq[:,targets.index(t)]>=.8).any());bestrows.append(r)
    best=pd.DataFrame(bestrows);best.to_csv(out/'leading_module_by_target.csv',index=False)
    anchors=pd.read_csv(WORK/'q2/top3_all_50_sensory_dimensions.csv')
    for r in anchors.itertuples():
        j=features.index(r.field);t=r.target
        mr=table[(table.threshold==.6)&(table.target==t)&(table.module==node.iloc[j].module)].iloc[0]
        targets_rows.append(dict(target=t,rank=r.rank,field=r.field,chemical_name=r.chemical_name,individual_frequency=r.bootstrap_top3_frequency,stratified_individual_frequency=r.stratified_top3_frequency,module=int(mr.module),module_union_frequency=float(mr.module_union_frequency),module_size=int(mr.eligible_module_size),largest_member_frequency=float(mr.largest_member_frequency),uniform_reference=float(mr.uniform_three_draw_reference),sign_agreement=float(r.sign_agreement),within_style_rho=float(r.within_style_rho)))
    at=pd.DataFrame(targets_rows);at.to_csv(out/'top3_with_module_context.csv',index=False)
    rows=[]
    for th,g in best.groupby('threshold'):
        rows.append(dict(threshold=th,targets=50,at_least_one_module_frequency_0_8=int((g.module_union_frequency>=.8).sum()),at_least_one_individual_frequency_0_8=int(g.any_individual_at_least_0_8.sum()),module_0_8_without_individual_0_8=int(((g.module_union_frequency>=.8)&~g.any_individual_at_least_0_8).sum()),mean_leading_module_size=float(g.eligible_module_size.mean())))
    summary={'bootstrap_replicates':400,'eligible_compounds':len(eligible),'reconstruction_max_frequency_difference':err,'graph_definition':'Full-data descriptive absolute Spearman graph and deterministic weighted modularity; fixed module membership per threshold. Bootstrap preprocessing/model were refitted in the original study.','module_frequency':'Fraction of stored bootstrap Top-3 sets that contain at least one eligible compound in the module; joint union, not sum of marginal frequencies.','uniform_reference':'Hypergeometric probability of at least one module member in 3 uniformly chosen distinct eligible compounds. Descriptive size reference, not a p value.','interpretation':'Substitution within correlated descriptive modules; not causal identification or stability of the graph partition itself.','threshold_summary':rows}
    dump(summary,out/'summary.json');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(best[(best.threshold==.6)&(best.module_union_frequency>=.8)&~best.any_individual_at_least_0_8].to_string(index=False),flush=True)

if __name__=='__main__':
    grading();modules()
