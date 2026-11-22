"""Read-only exact accounting of frozen OOF fusion; no fit or weight search."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT=HERE.parent/'fusion_replay_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def geometry(ea,eb,variance):
    a=float(np.mean(ea**2/variance));b=float(np.mean(eb**2/variance))
    c=float(np.mean(ea*eb/variance));d=float(np.mean((ea-eb)**2/variance))
    delta=abs(a-b);half=float(np.mean(((ea+eb)/2)**2/variance))
    # Analytic envelope only: no diagnostic weight is fitted or applied.
    envelope=max(d-delta,0.)**2/(4*d) if d>0 else 0.
    gain=min(a,b)-half
    return dict(y16_nmse=a,native50_nmse=b,uncentered_cross_moment=c,
                disagreement=d,endpoint_gap=delta,best_endpoint='y16' if a<=b else 'native50',
                half_nmse=half,disagreement_credit=d/4,endpoint_penalty=delta/2,
                half_gain_over_best=gain,half_gain_formula=d/4-delta/2,
                some_constant_convex_mix_improves=d>delta,
                half_improves=d>2*delta,
                empirical_constant_weight_envelope=envelope,
                half_fraction_of_envelope=gain/envelope if envelope>0 else np.nan,
                identity_error=abs(d-(a+b-2*c)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=DEFAULT/'outer_predictions.npz')
    parser.add_argument('--weights',type=Path,default=None,help='Frozen fitted_weights.csv; defaults beside predictions.')
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args();predictions=args.predictions.resolve();cache=predictions.parent
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    inputs=[predictions,args.weights.resolve()if args.weights else cache/'fitted_weights.csv'];before={str(p):sha(p)for p in inputs}
    with np.load(inputs[0],allow_pickle=False)as z:a={k:z[k]for k in z.files}
    y=a['Y'];ea=a['raw231_y16_no_residual']-y;eb=a['raw231_y50_native']-y
    v=y.var(0);assert y.shape==(250,50) and np.all(v>0)
    weights=pd.read_csv(inputs[1],float_precision='round_trip').set_index('outer_fold')
    assert set(np.unique(a['fold']))==set(weights.index)
    w=np.full_like(y,np.nan)
    for fold,row in weights.iterrows():
        w[a['fold']==fold,:30]=row.weight_core_y16
        w[a['fold']==fold,30:]=row.weight_other_y16
    assert np.isfinite(w).all() and ((w>=0)&(w<=1)).all()
    fixed={'fixed_half_y16_native50':np.full_like(w,.5),'inner_groupwise_y16_native50':w}
    max_prediction_error=0.
    for name,weight in fixed.items():
        pred=weight*a['raw231_y16_no_residual']+(1-weight)*a['raw231_y50_native']
        error=float(np.max(np.abs(pred-a[name])));assert error<1e-12
        max_prediction_error=max(max_prediction_error,error)
    groups={'all50':np.arange(50),'core30':np.arange(30),'other20':np.arange(30,50)}
    group=[];target=[];account=[];fold_account=[]
    for name,js in groups.items():
        g=geometry(ea[:,js],eb[:,js],v[js]);group.append(dict(targets=name,n=250,**g))
        for method,weight in fixed.items():
            for f in ['pooled',0,1,2,3,4]:
                ix=np.ones(len(y),bool)if f=='pooled' else a['fold']==f
                x=ea[ix][:,js];z=eb[ix][:,js];ww=weight[ix][:,js]
                la=float(np.mean(x*x/v[js]));lb=float(np.mean(z*z/v[js]));anchor=min(la,lb)
                endpoint=float(np.mean((ww*x*x+(1-ww)*z*z)/v[js]))
                credit=float(np.mean(ww*(1-ww)*(x-z)**2/v[js]))
                actual=float(np.mean((a[method][ix][:,js]-y[ix][:,js])**2/v[js]))
                row=dict(targets=name,method=method,fold=f,n=int(ix.sum()),best_endpoint='y16'if la<=lb else'native50',
                         best_endpoint_nmse=anchor,fusion_nmse=actual,disagreement_credit=credit,
                         signed_endpoint_penalty=endpoint-anchor,gain_over_best=anchor-actual,
                         accounted_gain=credit-(endpoint-anchor),identity_error=abs(actual-(endpoint-credit)))
                (account if f=='pooled'else fold_account).append(row)
    for j,name in enumerate(a['targets']):
        target.append(dict(target=str(name),group='core30'if j<30 else'other20',**geometry(ea[:,j:j+1],eb[:,j:j+1],v[j:j+1])))
    frames={'group_geometry.csv':pd.DataFrame(group),'per_target_geometry.csv':pd.DataFrame(target),
            'frozen_method_accounting.csv':pd.DataFrame(account),'fold_group_accounting.csv':pd.DataFrame(fold_account)}
    for name,frame in frames.items():
        payload=frame.to_csv(index=False,float_format='%.17g');tmp=out/(name+'.tmp');tmp.write_text(payload);os.replace(tmp,out/name)
        restored=pd.read_csv(out/name,float_precision='round_trip');assert len(restored)==len(frame)
        for column in frame:
            if frame[column].dtype.kind in 'fiu':
                np.testing.assert_allclose(restored[column],frame[column],rtol=0,atol=0,equal_nan=True)
    errors=[r['identity_error']for r in group+target+account+fold_account]
    gain_error=max(abs(r['half_gain_over_best']-r['half_gain_formula'])for r in group+target)
    assert max(errors)<1e-12 and gain_error<1e-12
    # Independent cross-check against existing frozen summaries, not this script's
    # own algebra. The summaries are never replaced and no learner is imported.
    checks=[]
    reference_half=cache/'exact_half_blend_accounting.csv'
    if reference_half.exists():
        ref=pd.read_csv(reference_half,float_precision='round_trip').set_index('targets')
        for row in group:
            old=ref.loc[row['targets']]
            for field,key in [('y16_nmse','y16_nmse'),('native50_nmse','native50_nmse'),('half_nmse','half_blend_nmse'),('disagreement','normalized_disagreement')]:
                checks.append(abs(row[field]-old[key]))
        before[str(reference_half)]=sha(reference_half)
    reference_metrics=cache/'pooled_metrics.csv'
    if reference_metrics.exists():
        ref=pd.read_csv(reference_metrics,float_precision='round_trip').set_index('method')
        for row in account:
            checks.append(abs((1-row['fusion_nmse'])-ref.loc[row['method'],row['targets']]))
        before[str(reference_metrics)]=sha(reference_metrics)
    assert max(checks,default=0.)<1e-12
    assert all(sha(Path(p))==value for p,value in before.items())
    counts={name:dict(targets=len(df),half_better_than_best=int(df.half_gain_over_best.gt(1e-12).sum()),
                      half_not_better=int(df.half_gain_over_best.le(1e-12).sum()),
                      some_constant_mix_improves=int(df.some_constant_convex_mix_improves.sum()),
                      only_nonhalf_mix_improves=int((df.some_constant_convex_mix_improves&~df.half_improves).sum()),
                      no_constant_mix_improves=int((~df.some_constant_convex_mix_improves).sum()))
            for name,df in pd.DataFrame(target).groupby('group')}
    record=dict(status='PASS',source_sha256=before,source_files_unchanged=True,
                new_model_fits=0,new_weight_fits=0,new_weight_searches=0,new_predictions=0,new_bootstrap_resamples=0,
                prediction_reconstruction_max_error=max_prediction_error,loss_identity_max_error=max(errors),
                half_gain_formula_max_error=gain_error,per_target_counts=counts,
                independent_existing_summary_checks=len(checks),independent_summary_max_error=max(checks,default=0.),
                variance='same pooled250 evaluation labels, ddof=0; diagnostics only; never used to refit weights',
                analytic_envelope_scope='One common constant convex weight within each evaluated group; not an upper bound for fold/group-varying fitted weights.',
                scope='Exact same-sample empirical accounting; not external generalization, causal or future-risk guarantee.',
                outputs={name:sha(out/name)for name in frames},script_sha256=sha(Path(__file__)))
    (out/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(group)[['targets','disagreement_credit','endpoint_penalty','half_gain_over_best','half_fraction_of_envelope']].to_string(index=False))
    print(pd.DataFrame(account)[['targets','method','disagreement_credit','signed_endpoint_penalty','gain_over_best']].to_string(index=False))
    print(json.dumps(counts))

if __name__=='__main__':main()
