"""Predict with complementary PCA16 and Native50 TabICLv2 branches.

Normal execution uses the pinned pretrained network and the 250-row training
context. --replay-cache is a separate exact-validation path using historical
outer-CV predictions; it performs no neural inference and is labelled as such.
"""
from pathlib import Path
import argparse,importlib.metadata,json,os,sys,time
from fusion_model import combine_predictions,relative_grades
from single_branch_interface import chemistry_input,check_context,sha

def replay(args,X,identifiers,features,targets,mode):
    import numpy as np
    with np.load(args.replay_cache,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    if str(c['cache_role'].item())!='saved_outer_CV_predictions_from_200_row_contexts_not_new_250_context_inference':
        raise ValueError('Unsupported cache provenance.')
    np.testing.assert_array_equal(c['features'],features);np.testing.assert_array_equal(c['targets'],targets)
    source={str(v):i for i,v in enumerate(c['beer'])}
    try:ix=np.asarray([source[str(v)] for v in identifiers],int)
    except KeyError as e:raise ValueError('A query identifier has no saved CV prediction: '+str(e)) from None
    np.testing.assert_allclose(X,c['query_X'][ix],rtol=0,atol=0,equal_nan=True,err_msg='Cache chemistry differs; cache replay cannot answer unseen or modified inputs.')
    a,b=c['prediction_y16'][ix],c['prediction_native50'][ix]
    p=np.empty_like(a);grades=np.empty(a.shape,int)
    for f in np.unique(c['fold'][ix]):
        m=c['fold'][ix]==f
        wc=wo=.5
        if mode=='inner_groupwise':wc=float(c['inner_core_weights'][int(f)]);wo=float(c['inner_other_weights'][int(f)])
        p[m]=combine_predictions(a[m],b[m],wc,wo)
        cuts=np.quantile(c['Y_reference'][c['fold']!=f],np.arange(1,7)/7,axis=0)
        grades[m]=relative_grades(p[m],cuts)
    return p,grades,a,b,{'execution':'historical_outer_cv_cache_replay','neural_inference_executed':False,'cache_sha256':sha(args.replay_cache),'cache_role':str(c['cache_role'].item()),'grade_reference':'Each archived row uses its own outer-training 200-row reference; not the final 250-row deployment cuts.','context_rows_used_for_predictions':200,'query_rows_verified_exactly':len(ix),'timings':[]}

def native_inference(args,a,X,manifest):
    import numpy as np
    for pkg,req in [('tabicl',manifest['required_tabicl_version']),('torch',manifest['required_torch_version'])]:
        try:actual=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:raise RuntimeError(f'{pkg} is unavailable. Install the pinned deployment requirements to run new neural inference; cache replay is a separate validation mode.') from None
        if actual!=req:raise RuntimeError(f'{pkg} version {actual} differs from pinned {req}.')
    checkpoint=args.checkpoint or (args.artifact/manifest['checkpoint']['relative_path']).resolve()
    if not checkpoint.is_file():raise FileNotFoundError('Pinned public TabICL checkpoint is absent; provide --checkpoint or run fetch_public_weights.py --model tabicl.')
    if sha(checkpoint)!=manifest['checkpoint']['sha256']:raise ValueError('Checkpoint identity differs from the original benchmark.')
    import torch
    from tabicl import TabICLRegressor
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads);torch.manual_seed(manifest['parameters']['random_state'])
    latent=np.empty((len(X),16));native=np.empty((len(X),50));timings=[]
    with threadpool_limits(limits=args.threads):
        for branch,target,destination in [('y16',a['context_targets'],latent),('native50',a['Y'],native)]:
            for j in range(target.shape[1]):
                # The pinned public API rebuilds the per-target scaler,
                # feature encoder and context; no cross-target cache reuse.
                model=TabICLRegressor(n_estimators=8,n_jobs=args.threads,device='cpu',use_amp=False,model_path=checkpoint,allow_auto_download=False,random_state=20261023,batch_size=8,kv_cache=False)
                start=time.perf_counter();model.fit(a['X'],target[:,j]);fit_s=time.perf_counter()-start
                start=time.perf_counter();destination[:,j]=model.predict(X);predict_s=time.perf_counter()-start
                timings.append({'branch':branch,'target':j,'fit_seconds':fit_s,'predict_seconds':predict_s})
                print(json.dumps(timings[-1]),file=sys.stderr,flush=True)
                del model
    y16=(latent@a['pca_components']+a['pca_mean'])*a['target_scale']+a['target_mean']
    w=manifest['fusion'];p=combine_predictions(y16,native,w['core30_y16_weight'],w['other20_y16_weight']);grades=relative_grades(p,a['relative_grade_cuts'])
    return p,grades,y16,native,{'execution':'new_250_context_neural_inference','neural_inference_executed':True,'checkpoint_sha256':sha(checkpoint),'scalar_inference_targets':66,'neural_weights_shared_across_fit_calls':False,'grade_reference':'All 250 training-context sensory rows; relative intensity only.','context_rows_used_for_predictions':250,'timings':timings}

def predict(args):
    for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]=str(args.threads)
    import numpy as np
    import pandas as pd
    manifest_path=args.artifact/'manifest.json';manifest=json.loads(manifest_path.read_text());context=args.artifact/manifest['context_file']
    if sha(context)!=manifest['context_sha256']:raise ValueError('Context identity mismatch.')
    with np.load(context,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    features,targets=check_context(a,manifest)
    frame,X,idcol,ids,missing,extra=chemistry_input(args.input,features,targets,args.missing_columns,args.id_column)
    validation={'valid':True,'rows':len(X),'chemical_features':231,'sensory_targets':50,'identifier':idcol,'missing_columns_filled':missing,'extra_columns_ignored':extra,'mode':manifest['fusion']['mode'],'context_rows':len(a['X']),'neural_inference_executed':False}
    if args.validate_only:print(json.dumps(validation,ensure_ascii=False));return
    if args.output.exists():raise FileExistsError('Refusing to overwrite an existing prediction file.')
    values=replay(args,X,ids,features,targets,manifest['fusion']['mode']) if args.replay_cache else native_inference(args,a,X,manifest)
    p,g,pa,pb,audit=values
    for value in [p,g,pa,pb]:
        if value.shape!=(len(X),50) or not np.isfinite(value).all():raise RuntimeError('Invalid output matrix.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def write(path,arr):
        d=pd.DataFrame(arr,columns=targets);d.insert(0,idcol,ids);d.to_csv(path,index=False);return path
    write(args.output,p);grade_path=write(args.output.with_name(args.output.stem+'.relative_grades_0_6.csv'),g)
    if args.write_branches:
        write(args.output.with_name(args.output.stem+'.y16.csv'),pa);write(args.output.with_name(args.output.stem+'.native50.csv'),pb)
    audit.update({'input_sha256':sha(args.input),'artifact_manifest_sha256':sha(manifest_path),'output_sha256':sha(args.output),'relative_grades_sha256':sha(grade_path),'output_units':'Original standardized sensory scale of attachment 4; grades are relative 0-6, not recovered panel scores.','deployment_mode':manifest['fusion']['mode'],'validation':validation,'downstream_transfer_validation_inherited':False})
    args.output.with_suffix(args.output.suffix+'.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({'output':str(args.output),'grades':str(grade_path),'rows':len(X),'execution':audit['execution'],'neural_inference_executed':audit['neural_inference_executed']},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parent/'deployment_fixed_half');p.add_argument('--checkpoint',type=Path);p.add_argument('--threads',type=int,default=2);p.add_argument('--id-column');p.add_argument('--missing-columns',choices=['error','nan'],default='error');p.add_argument('--validate-only',action='store_true');p.add_argument('--replay-cache',type=Path,help='Exact original CV-row validation only; never new-query inference.');p.add_argument('--write-branches',action='store_true');args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    predict(args)
