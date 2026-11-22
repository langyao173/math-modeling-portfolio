"""New exploratory diagnosis of a fixed classifier's Tripel region.

This does not alter the earlier target-distance optimization. It tests its
prespecified nested forward supports and all 225 coordinates. No sensory model
is retrained. Feasibility is conditional on the exact fitted classifier, support,
rank box, chemical caps and linear surrogate; it is not physical style proof.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linprog,minimize,Bounds,LinearConstraint
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
AWORK=HERE.parents[1]
BUDGET_DIR=HERE.parent/'budget_transfer'
sys.path.insert(0,str(AWORK/'q3'))
sys.path.insert(0,str(AWORK/'q1'))
from conditional_model import DERIVED,TECHNICAL
from transfer_profiles import DEFECTS,as_raw
from sensory_model import predict_bundle

BUDGETS=[3,5,10,20,40,60,80,100,140,180,225]
MARGINS=[0.,.025,.05,.1,.2,.4,.8]
ALPHAS=np.unique(np.r_[.01,.025,np.arange(1,21)/20.])
LAMBDA=.2
EPS=1e-7
TOL=1e-8
LP_OPTIONS={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def serial(obj):
    if isinstance(obj,(np.integer,np.floating)):return obj.item()
    if isinstance(obj,np.bool_):return bool(obj)
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj).__name__)

def main():
    started=time.perf_counter()
    data=Path(os.environ.get('MATHORCUP_DATA_DIR',AWORK.parent/'data/A')).expanduser().resolve()
    files={'data':data/'1.csv','calibrated':data/'8.xlsx',
      'frozen_q3':AWORK/'q3/transfer_and_augmentation_models.joblib',
      'frozen_q1':AWORK/'q1/sensory_ensemble.joblib',
      'reference_script':BUDGET_DIR/'run_budget_transfer.py',
      'support_definitions':BUDGET_DIR/'selected_supports.csv',
      'old_diagnostics':BUDGET_DIR/'selected_diagnostics.csv',
      'old_profiles':BUDGET_DIR/'selected_profiles.csv'}
    initial_hashes={k:sha256(v) for k,v in files.items()}
    protocol=dict(created_before_execution=True,new_exploratory_diagnostic=True,
      budgets=BUDGETS,margins=MARGINS,alphas=ALPHAS.tolist(),lambda_preservation=LAMBDA,
      rank_dimensions=225,style_constraint_epsilon=EPS,numerical_acceptance_tolerance=TOL,
      classifier=dict(estimator='LogisticRegression',C=.1,max_iter=3000,class_weight='balanced',training='all original observed 22 styles in the unchanged 225-dimensional rank map'),
      support_source='unaltered budget_transfer/selected_supports.csv; forward_schur_k* and all_225 only',
      chemical_caps='identical to expanded-budget script: observed direct field cannot increase; missing baseline is capped at threshold',
      linear_stale='same stored linear surrogate, a.T delta <= -m',
      nonlinear_stale='same frozen Q1 ensemble, evaluated after inverse map, exact frozen raw fields and source-anchored auxiliary reconstruction',
      qp='same original distance-plus-preservation objective, SLSQP maxiter=600 ftol=1e-10; feasible LP witness initializes each margin',
      selection='minimum original quadratic objective among accepted Tripel candidates on original alpha grid, retaining candidates from all smaller nested supports; unchanged source is an explicitly unsuccessful fallback',
      phase_one='If primary LP reports infeasible, solve minimum uniform inequality relaxation; save nonnegative dual weights and the box-minimum separation lower bound.',
      interpretation='first successful tested nested support is not a global minimum-field proof; fixed classifier region is not experimentally established beer style',
      inputs={k:str(v) for k,v in files.items()},input_sha256=initial_hashes)
    (HERE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    df=pd.read_csv(files['data']);original=df.set_index('beer')
    calibrated=pd.read_excel(files['calibrated'],sheet_name='Calibrated chemical dataset').set_index('beer')
    pack=joblib.load(files['frozen_q3']);control=pack['model'];features=list(pack['features']);proxy=pack['stale_proxy'];summaries=pack['derived_predictors']
    bundle=joblib.load(files['frozen_q1'])
    x=df[features].to_numpy(float);z=control.map_.transform(x);p=len(features)
    assert p==225
    mu=control.mean_['Tripel'];precision=np.linalg.inv(control.cov_['Tripel']);h=precision+LAMBDA*np.eye(p);a=proxy.coef_
    clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(z,df.tasting_category_fine)
    joblib.dump(clf,HERE/'frozen_style_classifier.joblib')
    ti=list(clf.classes_).index('Tripel');others=np.flatnonzero(np.arange(len(clf.classes_))!=ti)
    wdiff=clf.coef_[ti]-clf.coef_[others];bdiff=clf.intercept_[ti]-clf.intercept_[others]
    competitor_names=clf.classes_[others].tolist()
    supports=pd.read_csv(files['support_definitions'])
    supports=supports[supports.method.str.startswith('forward_schur_k')|supports.method.eq('all_225')].copy().reset_index(drop=True)
    supports['case_id']=np.arange(len(supports))
    supports.to_csv(HERE/'support_definitions.csv',index=False)
    assert len(supports)==121
    for beer,g in supports.groupby('beer'):
        prev=set()
        for r in g.sort_values('allowed_coordinates').itertuples():
            ss=set(json.loads(r.selected_fields));assert len(ss)==r.allowed_coordinates and prev<=ss;prev=ss
    basepred=predict_bundle(bundle,df);basepred.index=df.beer
    old_diag=pd.read_csv(files['old_diagnostics'])
    old_profiles=pd.read_csv(files['old_profiles'])
    mask=old_profiles.method.str.startswith('forward_schur_k')|old_profiles.method.eq('all_225')
    old_profiles=old_profiles.loc[mask].reset_index(drop=True)
    old_diag=old_diag.set_index(['beer','method']).loc[list(zip(old_profiles.beer,old_profiles.method))].reset_index()
    old_z=control.map_.transform(old_profiles[features].to_numpy(float))
    style_match=bool(np.array_equal(clf.predict(old_z),old_diag.predicted_style.to_numpy()))
    probability_max_difference=float(np.max(np.abs(clf.predict_proba(old_z)[:,ti]-old_diag.target_probability)))
    assert style_match and probability_max_difference<1e-10
    contexts={}
    for idx in np.flatnonzero(df.tasting_category_fine.to_numpy()=='Pils/Lager'):
        src=df.iloc[idx];beer=int(src.beer);z0=z[idx];x0=x[idx]
        lo=control.map_.lower_.copy();hi=control.map_.upper_.copy();caps={}
        for col,info in DEFECTS.items():
            j=features.index(col)
            cap=x0[j] if np.isfinite(x0[j]) else (np.log10(info['threshold']) if info['transform']=='log10' else info['threshold'])
            caprow=control.map_.impute(x0[None])[0];caprow[j]=cap
            hi[j]=min(hi[j],control.map_.transform(caprow[None])[0,j]);caps[j]=cap
        contexts[beer]=dict(index=int(idx),src=src,z0=z0,x0=x0,lo=lo,hi=hi,caps=caps,
             b=precision@(mu-z0),base_style_margin=wdiff@z0+bdiff,
             d0=float(np.sqrt((z0-mu)@precision@(z0-mu)/p)),stale0=float(basepred.loc[beer,'stale_hops']),
             original_objective=float(.5*(z0-mu)@precision@(z0-mu)),
             auxiliary0={c:float(reg.predict(z0[None])[0]) for c,reg in summaries.items()})
    lp_records=[];qp_records=[];witnesses=[];dual_weights=[];path_skips=[];profiles=[];path_z=[];path_meta=[]
    for case in supports.itertuples():
        beer=int(case.beer);ctx=contexts[beer];z0=ctx['z0'];x0=ctx['x0']
        sel=np.array([features.index(c) for c in json.loads(case.selected_fields)],int)
        frozen=np.ones(p,bool);frozen[sel]=False
        dlo=ctx['lo'][sel]-z0[sel];dhi=ctx['hi'][sel]-z0[sel]
        frozen_violation=max(float(np.max(ctx['lo'][frozen]-z0[frozen],initial=-np.inf)),float(np.max(z0[frozen]-ctx['hi'][frozen],initial=-np.inf)),0.)
        assert frozen_violation<=TOL
        A=np.vstack([a[sel],-wdiff[:,sel]])
        for margin in MARGINS:
            lp_id=len(lp_records);rhs=np.r_[-margin,ctx['base_style_margin']-EPS]
            ans=linprog(np.zeros(len(sel)),A_ub=A,b_ub=rhs,bounds=list(zip(dlo,dhi)),method='highs',options=LP_OPTIONS)
            rec=dict(lp_id=lp_id,case_id=int(case.case_id),beer=beer,method=case.method,allowed_coordinates=int(case.allowed_coordinates),margin=margin,
              solver_success=bool(ans.success),solver_status=int(ans.status),solver_message=str(ans.message),iterations=int(ans.nit),frozen_bound_violation=frozen_violation,
              lp_feasible=False,phase1_status=np.nan,phase1_min_relaxation=np.nan,dual_separation_lower_bound=np.nan,dual_weights_sum=np.nan)
            delta=np.full(p,np.nan);lam=np.full(A.shape[0],np.nan)
            if ans.x is not None:
                delta=np.zeros(p);delta[sel]=ans.x
                rec.update(max_lower_violation=float(np.max(dlo-ans.x)),max_upper_violation=float(np.max(ans.x-dhi)),max_inequality_violation=float(np.max(A@ans.x-rhs)),linear_stale_delta=float(a@delta),min_style_margin=float(np.min(ctx['base_style_margin']+wdiff@delta)))
                rec['lp_feasible']=bool(ans.success and max(rec['max_lower_violation'],rec['max_upper_violation'],rec['max_inequality_violation'])<=TOL)
            if not rec['lp_feasible']:
                phase=linprog(np.r_[np.zeros(len(sel)),1.],A_ub=np.column_stack([A,-np.ones(len(rhs))]),b_ub=rhs,bounds=list(zip(dlo,dhi))+[(0.,None)],method='highs',options=LP_OPTIONS)
                rec['phase1_status']=int(phase.status)
                if phase.success:
                    delta=np.zeros(p);delta[sel]=phase.x[:-1]
                    lam=np.maximum(-phase.ineqlin.marginals,0.)
                    if lam.sum()>0:lam/=lam.sum()
                    slope=lam@A
                    sep=float(np.sum(np.where(slope>=0,slope*dlo,slope*dhi))-lam@rhs)
                    rec.update(phase1_min_relaxation=float(phase.x[-1]),dual_separation_lower_bound=sep,dual_weights_sum=float(lam.sum()),
                      phase1_max_lower_violation=float(np.max(dlo-phase.x[:-1])),phase1_max_upper_violation=float(np.max(phase.x[:-1]-dhi)),
                      phase1_max_relaxed_inequality_violation=float(np.max(A@phase.x[:-1]-phase.x[-1]-rhs)),
                      phase1_unrelaxed_max_inequality_violation=float(np.max(A@phase.x[:-1]-rhs)),
                      phase1_min_style_margin=float(np.min(ctx['base_style_margin']+wdiff@delta)),phase1_linear_stale_delta=float(a@delta),
                      infeasibility_numerically_certified=bool(sep>1e-7))
            witnesses.append(delta);dual_weights.append(lam);lp_records.append(rec)
            if not rec['lp_feasible']:continue
            hs=h[np.ix_(sel,sel)];bs=ctx['b'][sel]
            begin=time.perf_counter()
            q=minimize(lambda d:float(.5*d@hs@d-bs@d),ans.x,jac=lambda d:hs@d-bs,method='SLSQP',bounds=Bounds(dlo,dhi),
              constraints=[LinearConstraint(A,-np.inf,rhs)],options={'maxiter':600,'ftol':1e-10})
            dd=np.zeros(p);dd[sel]=q.x
            violation=max(float(np.max(dlo-q.x)),float(np.max(q.x-dhi)),float(np.max(A@q.x-rhs)))
            ok=bool(q.success and violation<=TOL)
            qp_id=len(qp_records)
            qp_records.append(dict(qp_id=qp_id,lp_id=lp_id,case_id=int(case.case_id),beer=beer,method=case.method,allowed_coordinates=int(case.allowed_coordinates),margin=margin,
              success=ok,solver_success=bool(q.success),message=str(q.message),iterations=int(q.nit),seconds=time.perf_counter()-begin,
              max_constraint_violation=violation,linear_stale_delta=float(a@dd),min_style_margin=float(np.min(ctx['base_style_margin']+wdiff@dd)),
              objective_delta=float(.5*dd@h@dd-ctx['b']@dd)))
            if not ok:continue
            for alpha in ALPHAS:
                zz=z0+alpha*dd
                previolation=max(float(np.max(ctx['lo']-zz)),float(np.max(zz-ctx['hi'])))
                meta=dict(case_id=int(case.case_id),qp_id=qp_id,lp_id=lp_id,beer=beer,method=case.method,allowed_coordinates=int(case.allowed_coordinates),margin=margin,alpha=float(alpha))
                if previolation>TOL:
                    path_skips.append(dict(**meta,status='rank_box_violation_before_inverse',max_rank_box_violation=previolation))
                    continue
                xn=control.map_.inverse_transform(zz[None])[0];xn[frozen]=x0[frozen]
                for j,cap in ctx['caps'].items():
                    if j in sel:xn[j]=min(xn[j],cap)
                zz=control.map_.transform(xn[None])[0]
                row=ctx['src'].to_dict();row.update(dict(zip(features,xn)));row.update(meta)
                row.update(source_style='Pils/Lager',target_style='Tripel',new_exploratory_diagnostic=True)
                profiles.append(row);path_z.append(zz);path_meta.append(meta)
        print(f'beer {beer} {case.method}: LP cases {len(lp_records)}, feasible {sum(r["lp_feasible"] for r in lp_records)}, QP {len(qp_records)}, paths {len(profiles)}',flush=True)
    lp=pd.DataFrame(lp_records);qp=pd.DataFrame(qp_records)
    lp.to_csv(HERE/'linear_feasibility.csv',index=False);qp.to_csv(HERE/'qp_attempts.csv',index=False)
    pd.DataFrame(path_skips).to_csv(HERE/'path_grid_skips.csv',index=False)
    np.savez_compressed(HERE/'linear_witnesses_and_certificates.npz',delta=np.asarray(witnesses),dual_weights=np.asarray(dual_weights),competitors=np.asarray(competitor_names),features=np.asarray(features))
    candidates=pd.DataFrame(profiles);zz=np.asarray(path_z)
    for col,reg in summaries.items():
        new=reg.predict(zz)
        before=np.array([original.loc[int(b),col] for b in candidates.beer],float)
        anchor=np.array([contexts[int(b)]['auxiliary0'][col] for b in candidates.beer])
        candidates[col]=np.clip(np.where(np.isfinite(before),before+new-anchor,new),np.nanmin(df[col]),np.nanmax(df[col]))
    raw={c:10.**candidates[c].to_numpy(float) for c in ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']}
    candidates['kcalperc']=np.log10(5.523*raw['ethanol..v.v.']+.4*(raw['GFS']+raw['unfermentables']+raw['protein.g.L.'])+.0003*raw['lactic_acid.mg.L.'])
    pp=[]
    for start in range(0,len(candidates),512):pp.append(predict_bundle(bundle,candidates.iloc[start:start+512]))
    pred=pd.concat(pp,ignore_index=True)
    styles=clf.predict(zz);probs=clf.predict_proba(zz)[:,ti];scores=zz@wdiff.T+bdiff
    checks=[]
    for i,meta in enumerate(path_meta):
        beer=meta['beer'];ctx=contexts[beer];delta=zz[i]-ctx['z0'];row=candidates.iloc[i]
        chemical_violation=0.;unresolved=0
        for j,cap in ctx['caps'].items():
            v=row[features[j]]
            if not np.isfinite(v):unresolved+=1
            else:chemical_violation=max(chemical_violation,float(v-cap))
        lin=float(a@delta);nonlin=float(pred.loc[i,'stale_hops']-ctx['stale0']);stylemargin=float(scores[i].min())
        rankviol=max(float(np.max(ctx['lo']-zz[i])),float(np.max(zz[i]-ctx['hi'])))
        dnew=float(np.sqrt((zz[i]-mu)@precision@(zz[i]-mu)/p))
        obj=float(.5*(zz[i]-mu)@precision@(zz[i]-mu)+LAMBDA/2*(delta@delta))
        vals=row[features].to_numpy(float);old=ctx['x0'];same=(vals==old)|(np.isnan(vals)&np.isnan(old))
        accepted=bool(chemical_violation<=TOL and unresolved==0 and lin<=TOL and nonlin<=TOL and rankviol<=TOL and stylemargin>=EPS-TOL and styles[i]=='Tripel')
        checks.append(dict(**meta,candidate_index=i,source_target_distance=ctx['d0'],target_distance=dnew,target_distance_reduction=1-dnew/ctx['d0'],
          intervention_rms=float(np.linalg.norm(delta)/np.sqrt(p)),quadratic_objective=obj,rank_coordinates_changed=int((np.abs(delta)>TOL).sum()),
          raw_direct_fields_changed=int((~same).sum()),max_chemical_violation=chemical_violation,unresolved_missing_defects=unresolved,
          max_rank_box_violation=rankviol,linear_stale_delta=lin,ensemble_stale_delta=nonlin,
          min_style_margin=stylemargin,style_margin_violation=max(EPS-stylemargin,0.),predicted_style=str(styles[i]),target_probability=float(probs[i]),
          tripel_region_valid=bool(stylemargin>=EPS-TOL and styles[i]=='Tripel'),
          chemical_and_both_stale_valid=bool(chemical_violation<=TOL and unresolved==0 and lin<=TOL and nonlin<=TOL and rankviol<=TOL),accepted=accepted))
    checks=pd.DataFrame(checks);checks.to_csv(HERE/'candidate_checks.csv',index=False)
    selected_records=[];selected_profiles=[];selected_predictions=[];first_records=[]
    for case in supports.itertuples():
        ctx=contexts[int(case.beer)];k=int(case.allowed_coordinates)
        eligible=checks[(checks.beer==case.beer)&(checks.allowed_coordinates<=k)&checks.accepted]
        if len(eligible):
            best=eligible.sort_values(['quadratic_objective','raw_direct_fields_changed','margin','alpha'],kind='stable').iloc[0]
            ii=int(best.candidate_index);d=best.to_dict();row=candidates.iloc[ii].to_dict();prediction=pred.iloc[ii].to_dict()
            d.update(selection_origin_method=d['method'],method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),source_fallback=False)
            row.update(selection_origin_method=row['method'],method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),source_fallback=False,predicted_style=d['predicted_style'])
        else:
            z0=ctx['z0'];d=dict(beer=int(case.beer),method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),selection_origin_method='unchanged_source',source_fallback=True,
               candidate_index=-1,margin=0.,alpha=0.,source_target_distance=ctx['d0'],target_distance=ctx['d0'],target_distance_reduction=0.,intervention_rms=0.,
               quadratic_objective=ctx['original_objective'],rank_coordinates_changed=0,raw_direct_fields_changed=0,max_chemical_violation=0.,unresolved_missing_defects=sum(not np.isfinite(ctx['x0'][j]) for j in ctx['caps']),
               max_rank_box_violation=max(float(np.max(ctx['lo']-z0)),float(np.max(z0-ctx['hi']))),linear_stale_delta=0.,ensemble_stale_delta=0.,
               min_style_margin=float(ctx['base_style_margin'].min()),style_margin_violation=max(EPS-float(ctx['base_style_margin'].min()),0.),
               predicted_style=str(clf.predict(z0[None])[0]),target_probability=float(clf.predict_proba(z0[None])[0,ti]),tripel_region_valid=False,chemical_and_both_stale_valid=False,accepted=False)
            row=ctx['src'].to_dict();row.update(beer=int(case.beer),method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),selection_origin_method='unchanged_source',source_fallback=True,source_style='Pils/Lager',target_style='Tripel',predicted_style=d['predicted_style'],new_exploratory_diagnostic=True)
            prediction=basepred.loc[int(case.beer)].to_dict()
        selected_records.append(d);selected_profiles.append(row);selected_predictions.append(dict(beer=int(case.beer),method=case.method,**prediction))
    selected=pd.DataFrame(selected_records);final_profiles=pd.DataFrame(selected_profiles)
    selected.to_csv(HERE/'selected_diagnostics.csv',index=False);final_profiles.to_csv(HERE/'selected_profiles.csv',index=False);pd.DataFrame(selected_predictions).to_csv(HERE/'selected_sensory_predictions.csv',index=False)
    for beer in sorted(contexts):
        glp=lp[(lp.beer==beer)&lp.lp_feasible];gq=qp[(qp.beer==beer)&qp.success];gs=selected[(selected.beer==beer)&selected.accepted]
        prior=old_diag[(old_diag.beer==beer)&old_diag.predicted_style.eq('Tripel')]
        first_records.append(dict(beer=beer,first_tested_linear_feasible_budget=(int(glp.allowed_coordinates.min()) if len(glp) else None),
           first_tested_qp_success_budget=(int(gq.allowed_coordinates.min()) if len(gq) else None),
           first_tested_complete_acceptance_budget=(int(gs.allowed_coordinates.min()) if len(gs) else None),
           old_distance_objective_first_tested_tripel_budget=(int(prior.allowed_coordinates.min()) if len(prior) else None),
           has_complete_candidate=bool(len(gs)),interpretation='first success on fixed nested support grid, not globally minimum fields'))
    first=pd.DataFrame(first_records);first.to_csv(HERE/'first_tested_budgets.csv',index=False)
    first_keys={(int(r.beer),int(r.first_tested_complete_acceptance_budget)) for r in first.itertuples() if pd.notna(r.first_tested_complete_acceptance_budget)}
    chosen=final_profiles.apply(lambda r:(int(r.beer),int(r.allowed_coordinates)) in first_keys,axis=1)
    final_profiles.loc[chosen].to_csv(HERE/'first_accepted_profiles.csv',index=False)
    selected.loc[chosen].to_csv(HERE/'first_accepted_diagnostics.csv',index=False)
    oldcols=['beer','method','allowed_coordinates','target_distance_reduction','intervention_rms','quadratic_objective','predicted_style','target_probability','ensemble_stale_delta']
    comparison=selected.merge(old_diag[oldcols].rename(columns={c:'old_'+c for c in oldcols if c not in ['beer','method','allowed_coordinates']}),on=['beer','method','allowed_coordinates'],validate='one_to_one')
    comparison['new_minus_old_distance_reduction']=comparison.target_distance_reduction-comparison.old_target_distance_reduction
    comparison['new_minus_old_objective']=comparison.quadratic_objective-comparison.old_quadratic_objective
    comparison.to_csv(HERE/'comparison_with_distance_objective.csv',index=False)
    certificates=[];changes=[];frozen_ok=True;technical_ok=True
    support_lookup={(r.beer,r.method):set(json.loads(r.selected_fields)) for r in supports.itertuples()}
    for row in final_profiles.itertuples(index=False,name=None):
        row=dict(zip(final_profiles.columns,row));beer=int(row['beer']);src=original.loc[beer];support=support_lookup[beer,row['method']]
        for col in features:
            before=src[col];after=row[col];same=bool((pd.isna(before) and pd.isna(after)) or before==after)
            if col not in support and not same:frozen_ok=False
            if not same:changes.append(dict(beer=beer,method=row['method'],chemical=col,before=before,after=after,selected=col in support,baseline_observed=bool(np.isfinite(before))))
        for tech in TECHNICAL:
            technical_ok &= bool((pd.isna(src[tech]) and pd.isna(row[tech])) or src[tech]==row[tech])
        for col,info in DEFECTS.items():
            before=as_raw(src[col],info);after=as_raw(row[col],info);original_cal=calibrated.loc[beer,col] if col in calibrated else np.nan
            status='missing' if not np.isfinite(src[col]) else ('zero_or_nonpositive_replacement' if info['transform']=='log10' and (not np.isfinite(original_cal) or original_cal<=0) else 'positive_reported_measurement')
            certificates.append(dict(beer=beer,method=row['method'],chemical=col,unit=info['unit'],source_status=status,source_representative=before,after_representative=after,threshold=info['threshold'],
                nonincrease=(bool(after<=before+TOL) if np.isfinite(before) and np.isfinite(after) else None),output_below_threshold=(bool(after<=info['threshold']+TOL) if np.isfinite(after) else None),source_fallback=row['source_fallback']))
    pd.DataFrame(changes).to_csv(HERE/'changed_direct_fields.csv',index=False);pd.DataFrame(certificates).to_csv(HERE/'chemical_certificate.csv',index=False)
    assert frozen_ok and technical_ok
    assert np.all(selected.raw_direct_fields_changed<=selected.allowed_coordinates)
    groups=[]
    for k in BUDGETS:
        glp=lp[(lp.allowed_coordinates==k)&lp.margin.eq(0.)];g=selected[selected.allowed_coordinates==k];gg=g[g.accepted];old=old_diag[old_diag.allowed_coordinates==k]
        groups.append(dict(budget=k,n_sources=len(g),linear_feasible_m0=int(glp.lp_feasible.sum()),complete_accepted=int(g.accepted.sum()),source_fallbacks=int(g.source_fallback.sum()),
             accepted_mean_distance_reduction=float(gg.target_distance_reduction.mean()) if len(gg) else None,all_source_mean_distance_reduction=float(g.target_distance_reduction.mean()),
             accepted_mean_intervention_rms=float(gg.intervention_rms.mean()) if len(gg) else None,accepted_mean_target_probability=float(gg.target_probability.mean()) if len(gg) else None,
             accepted_max_stale_delta=float(gg.ensemble_stale_delta.max()) if len(gg) else None,old_tripel_classified=int(old.predicted_style.eq('Tripel').sum()),old_mean_distance_reduction=float(old.target_distance_reduction.mean())))
    summary_table=pd.DataFrame(groups);summary_table.to_csv(HERE/'budget_summary.csv',index=False)
    final_hashes={k:sha256(v) for k,v in files.items()};assert initial_hashes==final_hashes
    summary=dict(n_sources=len(contexts),n_support_cases=len(supports),n_lp=len(lp),n_linear_feasible=int(lp.lp_feasible.sum()),n_infeasible=int((~lp.lp_feasible).sum()),
      n_numerically_certified_infeasible=int(lp.get('infeasibility_numerically_certified',pd.Series(dtype=bool)).fillna(False).sum()),
      n_qp=len(qp),n_qp_success=int(qp.success.sum()),n_path_profiles=len(candidates),n_preinverse_grid_skips=len(path_skips),n_accepted_paths=int(checks.accepted.sum()),
      n_sources_with_complete_candidate=int(first.has_complete_candidate.sum()),classifier_matches_old_labels=style_match,classifier_max_probability_difference=probability_max_difference,
      exact_frozen_direct_values_preserved=frozen_ok,technical_internal_standard_preserved=technical_ok,old_input_files_unchanged=initial_hashes==final_hashes,
      largest_accepted_rank_violation=float(checks.loc[checks.accepted,'max_rank_box_violation'].max()) if checks.accepted.any() else None,
      largest_accepted_stale_delta=float(checks.loc[checks.accepted,'ensemble_stale_delta'].max()) if checks.accepted.any() else None,
      minimum_accepted_style_margin=float(checks.loc[checks.accepted,'min_style_margin'].min()) if checks.accepted.any() else None,
      seconds=time.perf_counter()-started,evidence_scope=protocol['interpretation'])
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=serial))
    print(summary_table.to_string(index=False),flush=True)
    print(first.to_string(index=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False,default=serial),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
