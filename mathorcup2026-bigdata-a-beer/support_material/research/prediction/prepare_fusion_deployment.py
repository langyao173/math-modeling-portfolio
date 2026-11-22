"""Export existing 250-row context plus fixed or training-OOF calibrated fusion.

No neural model is loaded and no neural prediction is generated. Deploying
fixed-half is an engineering simplicity choice; the preregistered internal
primary experiment remains the two-group, train-only weighted procedure.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
from fusion_model import optimal_weight
from single_branch_interface import check_context,sha
ROOT=Path(__file__).resolve().parent

def main(args):
    manifest=json.loads((args.source_artifact/'manifest.json').read_text())
    source=args.source_artifact/manifest['context_file']
    if sha(source)!=manifest['context_sha256']:raise ValueError('Source context identity mismatch.')
    with np.load(source,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    features,targets=check_context(a,manifest)
    assert a['X'].shape==(250,231) and a['context_targets'].shape==(250,16)
    encoded=((a['Y']-a['target_mean'])/a['target_scale']-a['pca_mean'])@a['pca_components'].T
    np.testing.assert_allclose(a['context_targets'],encoded,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(a['relative_grade_cuts'],np.quantile(a['Y'],np.arange(1,7)/7,axis=0),rtol=0,atol=1e-12)
    wc=wo=.5;calibration='No weight fitting; fixed half on all targets.'
    with np.load(args.modern/'oof_arrays.npz',allow_pickle=False) as z:oof={k:z[k] for k in z.files}
    for k in ['Y','beer','targets']:np.testing.assert_array_equal(a[k],oof[k])
    if args.mode=='inner_groupwise':
        wc=optimal_weight(a['Y'],oof['raw231_y16_no_residual'],oof['raw231_y50_native'],np.arange(30))
        wo=optimal_weight(a['Y'],oof['raw231_y16_no_residual'],oof['raw231_y50_native'],np.arange(30,50))
        calibration='Final calibration on all 250 training OOF rows only; these deployment weights were not used to calculate any outer-CV performance.'
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Choose an empty artifact directory.')
    args.output.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,args.output/'model_context.npz')
    new=dict(manifest)
    new.update(format_version=2,model_family='TabICLv2 complementary-output fusion',chosen_cv_candidate=args.mode+'_y16_native50',context_file='model_context.npz',context_sha256=sha(args.output/'model_context.npz'),checkpoint={**manifest['checkpoint'],'relative_path':'../models/tabicl-regressor-v2-20260212.ckpt'},fusion={'mode':args.mode,'core30_y16_weight':wc,'other20_y16_weight':wo,'calibration':calibration,'branches':{'y16':'16 scalar latent targets, decoded to 50 responses','native50':'50 scalar original-response targets'},'scalar_inference_targets':66,'pretrained_neural_checkpoints':1},selection={'mode':'engineering_fixed_half' if args.mode=='fixed_half' else 'train_OOF_final_calibration','primary_internal_experiment':'inner_groupwise_y16_native50 remains the preregistered primary; all three prespecified procedures are reported.','choice_rationale':('Fixed-half deployment reduces calibration complexity; it is not claimed to optimize every metric.' if args.mode=='fixed_half' else 'Refit the prespecified two-group convex fusion rule on all training OOF rows for deployment; the resulting final weights are not used to recompute outer-CV scores.'),'outer_CV_predictions_recomputed_with_final_weights':False},inference_script_sha256=sha(ROOT/'predict_fusion_csv.py'),fit_predictions_generated=False,new_neural_inference_validated=False)
    (args.output/'manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2))
    import pandas as pd
    fitted=pd.read_csv(args.fusion_results/'fitted_weights.csv').sort_values('outer_fold')
    cache={'inner_core_weights':fitted.weight_core_y16.to_numpy(),'inner_other_weights':fitted.weight_other_y16.to_numpy(),'query_X':a['X'],'beer':a['beer'],'features':a['features'],'targets':a['targets'],'Y_reference':a['Y'],'fold':oof['fold'],'prediction_y16':oof['raw231_y16_no_residual'],'prediction_native50':oof['raw231_y50_native'],'cache_role':np.asarray('saved_outer_CV_predictions_from_200_row_contexts_not_new_250_context_inference')}
    np.savez_compressed(args.output/'outer_cv_validation_cache.npz',**cache)
    print(json.dumps({'output':str(args.output),'mode':args.mode,'weights':[wc,wo],'context_rows':250,'new_neural_inference':False},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-artifact',type=Path,required=True);p.add_argument('--modern',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fusion-results',type=Path,default=ROOT/'fusion_replay_v1');p.add_argument('--mode',choices=['fixed_half','inner_groupwise'],default='fixed_half');main(p.parse_args())
