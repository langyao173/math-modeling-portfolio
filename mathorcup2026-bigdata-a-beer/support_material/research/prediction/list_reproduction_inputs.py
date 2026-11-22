"""List the exact lightweight external files needed by prediction replay.

Paths in the manifest are relative to the portable deep/ package root.
The script inventories but never copies or changes source project files.
"""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main(args):
    deep=args.deep_root.resolve();m=Path('project/MathorCup_A/frontier_revision/experiments/modern_prediction');n=Path('checkpoint/unpacked/nested_selection')
    needs={}
    def add(path,*roles):needs.setdefault(str(path),set()).update(roles)
    add(m/'oof_arrays.npz','fusion_replay','independent_audit','deployment_reexport')
    add(n/'nested_oof.npz','fusion_replay','independent_audit')
    for outer in range(5):
        add(n/f'outer_{outer}/inner_oof.npz','fusion_replay','independent_audit')
        add(n/f'outer_{outer}/selection.json','fusion_replay')
        for view in ['raw231_y16','raw231_y50']:
            add(m/'checkpoints'/f'{view}_fold{outer}_complete.npz','fusion_replay','independent_audit')
            for inner in range(3):add(n/f'outer_{outer}/inner_{inner}/{view}_complete.npz','independent_audit')
    required=[]
    for label,roles in sorted(needs.items()):
        p=deep/label
        if not p.is_file():raise FileNotFoundError(p)
        required.append({'path':label,'required_by':sorted(roles),'bytes':p.stat().st_size,'sha256':sha(p)})
    optional=[{'path':'project/MathorCup_A/data/A/1.csv','required_by':['style_group_diagnostics'],'bytes':((deep/'project/MathorCup_A/data/A/1.csv').stat().st_size if (deep/'project/MathorCup_A/data/A/1.csv').is_file() else None)}, {'path':str(m/'final_artifact/manifest.json'),'required_by':['deployment_reexport']}, {'path':str(m/'final_artifact/model_context.npz'),'required_by':['deployment_reexport']}]
    # These are small provenance records. The replay/audit programs do not
    # consume them, but retaining them documents the original completed run.
    recommended=[str(n/'experiment_spec.json'),str(n/'results.json'),str(n/'inner_candidate_scores.csv'),str(m/'experiment_spec.json')]
    report={'path_base':'deep/','tested_python':'3.12.14','replay_dependencies':'prediction/requirements_replay.txt','external_required_files':required,'external_required_file_count':len(required),'external_required_bytes':sum(x['bytes'] for x in required),'optional_external_for_additional_commands':optional,'recommended_small_provenance_records':recommended,'deployment_validation_external_inputs':[],'deployment_validation_internal_inputs':['prediction/fusion_replay_v1/outer_predictions.npz','prediction/deployment_fixed_half/model_context.npz','prediction/deployment_fixed_half/manifest.json','prediction/deployment_fixed_half/outer_cv_validation_cache.npz','prediction/deployment_groupwise/model_context.npz','prediction/deployment_groupwise/manifest.json','prediction/deployment_groupwise/outer_cv_validation_cache.npz','prediction/validation_inputs/chemistry_250.csv'],'optional_neural_checkpoint':{'canonical_path':'prediction/models/tabicl-regressor-v2-20260212.ckpt','bytes':114324594,'sha256':'0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a','required_for':['new_neural_inference_only'],'alternative':'Pass --checkpoint PATH if stored elsewhere.'},'not_required':['torch/tabicl libraries embedded in the package','TabDPT safetensors or weights','A_work/q1/*.joblib classical fitted models','other modern input-rank checkpoint views','nested rank_pca64_y16 inner checkpoints','nested partial checkpoints or logs','any new model training or paid API'],'new_neural_forward_executed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'manifest':str(args.output),'required_files':len(required),'required_MiB':report['external_required_bytes']/1048576},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deep-root',type=Path,default=ROOT.parent);p.add_argument('--output',type=Path,default=ROOT/'REPRO_INPUTS.json');main(p.parse_args())
