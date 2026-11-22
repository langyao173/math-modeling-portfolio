"""Train-only low-dimensional fusion and hard-selection diagnostics.

All fits use each outer fold's own saved inner OOF archive. The global OOF
archive is used only to read the fixed outer predictions and evaluate them.
The result is internal nested-CV research, not a new external validation.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse, hashlib, json, os, re
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
import pandas as pd

A='raw231_y16_no_residual'
B='raw231_y50_native'
OLD='old_nested_ensemble'
GROUPS={'all50':np.arange(50),'core30':np.arange(30),'other20':np.arange(30,50)}
SEED=20261025
ROOT=Path(__file__).resolve().parent

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def eq(a,b,label):
    if not np.array_equal(a,b):raise ValueError('Identity/order mismatch: '+label)

def assess(y,p):
    if y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('Invalid predictions')
    den=((y-y.mean(0))**2).sum(0)
    if np.any(den<=0):raise ValueError('Constant response')
    r=1-((y-p)**2).sum(0)/den
    return {g:float(r[js].mean()) for g,js in GROUPS.items()}

def convex_weight(y,a,b,js):
    v=np.maximum(y.var(0),1e-12)
    d=a[:,js]-b[:,js]
    c=((b[:,js]-y[:,js])*d/v[js]).mean(axis=1)
    h=(d*d/v[js]).mean(axis=1)
    return float(np.clip(-c.mean()/h.mean(),0,1)) if h.mean()>1e-15 else 0.5, c,h

def main(args):
    proto=ROOT/'frozen_protocol.json'
    out=args.output
    if out.exists():raise FileExistsError('Refusing to overwrite an existing result directory')
    oof=load(args.modern/'oof_arrays.npz')
    Y,fold=oof['Y'],oof['fold']
    assert Y.shape==(250,50) and np.isfinite(Y).all()
    assert set(fold)==set(range(5))
    nested=load(args.nested/'nested_oof.npz')
    for key in ['Y','beer','fold','targets']:eq(oof[key],nested[key],'nested/'+key)
    original={k:oof[k] for k in [A,B,OLD]}
    original['tabicl_nested_selection']=nested['prediction']
    methods={k:np.full_like(Y,np.nan) for k in ['fixed_half_y16_native50','inner_core_scalar_y16_native50','inner_groupwise_y16_native50']}
    rows=[];diag=[];instability=[];sources={}
    candidate_order=None
    for f in range(5):
        tr,te=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        path=args.nested/f'outer_{f}'/'inner_oof.npz'
        inner=load(path)
        eq(inner['train_indices'],tr,f'{f}/train_indices')
        eq(inner['Y'],Y[tr],f'{f}/inner Y')
        eq(inner['beer'],oof['beer'][tr],f'{f}/inner beer')
        eq(inner['targets'],oof['targets'],f'{f}/targets')
        assert np.intersect1d(tr,te).size==0
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with (args.nested/f'outer_{f}'/'selection.json').open() as h:saved_selection=json.load(h)
        names=[x['model'] for x in saved_selection['candidate_scores']]
        if candidate_order is None:candidate_order=names
        else:assert names==candidate_order
        for n in names:
            assert inner[n].shape==(200,50) and np.isfinite(inner[n]).all()
        # Read the exact original checkpoint and verify that test rows align.
        for v,keys in [('raw231_y16',['prediction_no_residual']),('raw231_y50',['prediction_native'])]:
            a=load(args.modern/'checkpoints'/f'{v}_fold{f}_complete.npz')
            eq(a['train_indices'],tr,v+' tr');eq(a['test_indices'],te,v+' te')
            k=keys[0];name=v+'_'+k.removeprefix('prediction_')
            eq(a[k],oof[name][te],name+' outer')
        inner_y=inner['Y'];a,b=inner[A],inner[B]
        wc,cc,hc=convex_weight(inner_y,a,b,GROUPS['core30'])
        wo,co,ho=convex_weight(inner_y,a,b,GROUPS['other20'])
        methods['fixed_half_y16_native50'][te]=0.5*(oof[A][te]+oof[B][te])
        methods['inner_core_scalar_y16_native50'][te]=wc*oof[A][te]+(1-wc)*oof[B][te]
        gp=np.empty_like(Y[te]);gp[:,:30]=wc*oof[A][te,:30]+(1-wc)*oof[B][te,:30];gp[:,30:]=wo*oof[A][te,30:]+(1-wo)*oof[B][te,30:]
        methods['inner_groupwise_y16_native50'][te]=gp
        # This block only audits prior hard selection. No diagnostic below
        # can change any of the new predictions assigned above.
        candidate_scores={n:assess(inner_y,inner[n]) for n in names}
        chosen=sorted(names,key=lambda n:(-candidate_scores[n]['core30'],-candidate_scores[n]['all50'],n))[0]
        assert chosen==saved_selection['selected_model']
        eq(nested['prediction'][te],oof[chosen][te],'hard selected outer predictions')
        rng=np.random.default_rng(SEED+f)
        ids=rng.integers(200,size=(1000,200))
        v=inner_y.var(0)
        losses=np.stack([(((inner[n]-inner_y)**2)/v)[:,:30].mean(1) for n in names])
        resampled=losses[:,ids].mean(2)
        winners=np.argmin(resampled,axis=0)
        weights_core=np.clip(-cc[ids].mean(1)/hc[ids].mean(1),0,1)
        weights_other=np.clip(-co[ids].mean(1)/ho[ids].mean(1),0,1)
        rows.append({'outer_fold':f,'inner_rows':200,'outer_rows':50,'weight_core_y16':wc,'weight_other_y16':wo,'core_loss_curvature':float(hc.mean()),'other_loss_curvature':float(ho.mean()),'core_weight_conditional_q025':float(np.quantile(weights_core,.025)),'core_weight_conditional_q975':float(np.quantile(weights_core,.975)),'other_weight_conditional_q025':float(np.quantile(weights_other,.025)),'other_weight_conditional_q975':float(np.quantile(weights_other,.975)),'hard_selected':chosen})
        for ni,n in enumerate(names):
            instability.append({'outer_fold':f,'candidate':n,'inner_bootstrap_win_frequency':float(np.mean(winners==ni)),'hard_selected':n==chosen,'interpretation':'Fixed-inner-OOF conditional row bootstrap; no base-model refit'})
        for g,js in GROUPS.items():
            ia,ib=candidate_scores[A][g],candidate_scores[B][g]
            oa,ob=assess(Y[te],oof[A][te])[g],assess(Y[te],oof[B][te])[g]
            best_outer=max(names,key=lambda n:assess(Y[te],oof[n][te])[g])
            diag.append({'outer_fold':f,'targets':g,'inner_y16_r2':ia,'inner_native50_r2':ib,'inner_y16_minus_native50':ia-ib,'outer_y16_r2':oa,'outer_native50_r2':ob,'outer_y16_minus_native50':oa-ob,'inner_outer_sign_agree':bool((ia-ib)*(oa-ob)>0),'hard_selected':chosen,'hard_selected_outer_r2':assess(Y[te],oof[chosen][te])[g],'hindsight_outer_winner_DIAGNOSTIC_ONLY':best_outer,'hindsight_regret_DIAGNOSTIC_ONLY':assess(Y[te],oof[best_outer][te])[g]-assess(Y[te],oof[chosen][te])[g]})
    predictions={**original,**methods}
    for name,p in predictions.items():assert p.shape==Y.shape and np.isfinite(p).all(),name
    out.mkdir(parents=True)
    (out/'frozen_protocol.json').write_bytes(proto.read_bytes())
    pd.DataFrame(rows).to_csv(out/'fitted_weights.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'inner_outer_selection_diagnostics.csv',index=False)
    pd.DataFrame(instability).to_csv(out/'conditional_inner_selection_stability.csv',index=False)
    metrics=pd.DataFrame([{'method':n,**assess(Y,p)} for n,p in predictions.items()])
    metrics.to_csv(out/'pooled_metrics.csv',index=False)
    pd.DataFrame([{'method':n,'outer_fold':f,**assess(Y[fold==f],p[fold==f])} for n,p in predictions.items() for f in range(5)]).to_csv(out/'fold_metrics.csv',index=False)
    np.savez_compressed(out/'outer_predictions.npz',Y=Y,beer=oof['beer'],fold=fold,targets=oof['targets'],**predictions)
    # Exact squared-error accounting: half-blend loss equals average endpoint
    # loss minus one-quarter of normalized squared disagreement.
    variance=Y.var(0);account=[]
    for g,js in GROUPS.items():
        la=(((original[A]-Y)**2)/variance)[:,js].mean();lb=(((original[B]-Y)**2)/variance)[:,js].mean();lh=(((methods['fixed_half_y16_native50']-Y)**2)/variance)[:,js].mean();dis=(((original[A]-original[B])**2)/variance)[:,js].mean()
        assert abs(lh-(.5*(la+lb)-.25*dis))<1e-12
        account.append({'targets':g,'y16_nmse':la,'native50_nmse':lb,'half_blend_nmse':lh,'normalized_disagreement':dis,'diversity_gain_over_average_loss':.25*dis,'half_blend_delta_vs_native50_r2':lb-lh,'identity_error':lh-(.5*(la+lb)-.25*dis)})
    pd.DataFrame(account).to_csv(out/'exact_half_blend_accounting.csv',index=False)
    # All preregistered procedures and reference pairs are reported, including
    # negative/uncertain results. Bootstrap never changes predictions.
    refs=[A,B,OLD,'tabicl_nested_selection'];new=list(methods)
    error={n:(p-Y)**2 for n,p in predictions.items()}
    boot={};rng=np.random.default_rng(SEED+20000)
    for name in new:
        for ref in refs:boot[(name,ref)]=[]
    for _ in range(2000):
        ii=rng.integers(len(Y),size=len(Y));var=Y[ii].var(0)
        for name,ref in boot:
            delta=(error[ref][ii].mean(0)-error[name][ii].mean(0))/var
            boot[(name,ref)].append([float(delta[js].mean()) for js in GROUPS.values()])
    comparisons=[]
    for (name,ref),draws in boot.items():
        draws=np.asarray(draws);point={g:assess(Y,predictions[name])[g]-assess(Y,predictions[ref])[g] for g in GROUPS}
        for j,g in enumerate(GROUPS):comparisons.append({'method':name,'reference':ref,'targets':g,'delta_r2':point[g],'conditional_ci025':float(np.quantile(draws[:,j],.025)),'conditional_ci975':float(np.quantile(draws[:,j],.975)),'replications':2000,'refit':False,'external_independent_test':False})
    pd.DataFrame(comparisons).to_csv(out/'all_prespecified_comparisons.csv',index=False)
    metadata={'complete':True,'protocol_sha256':hashlib.sha256(proto.read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_modern_oof_sha256':hashlib.sha256((args.modern/'oof_arrays.npz').read_bytes()).hexdigest(),'inner_sources':sources,'new_paid_calls':0,'new_base_model_fits':0,'new_tabicl_inferences':0,'new_scalar_blend_weight_fits':10,'outer_labels_used_to_fit_blend':False,'untouched_external_validation':False,'all_new_methods_reported':new}
    (out/'run_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(metrics.to_string(index=False));print(json.dumps(metadata,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modern',type=Path,required=True);p.add_argument('--nested',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
