# Team review and independent work must be documented truthfully.
"""Re-evaluate the existing six-family pipeline on the author's published split.

This is a split-aligned, strictly preprocessed comparison, not a rerun of the
paper's full 10-model hyperparameter searches. The previous pipeline's grids,
loss and convex blending are unchanged; only its inner folds increase 3 -> 5.
The source CSV row order is retained, with 70/30 style-stratified seed-0 split.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, argparse
from pathlib import Path
from project_paths import DATA_HELP, resolve_data_dir
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'q1'))
from sensory_model import SensoryEstimator, specifications, learn_weights
from run_q1 import FAMILIES, assess, jsonable


def strict_jsonable(value):
    """Represent undefined constant-prediction correlations as JSON null."""
    if isinstance(value, dict):
        return {k: strict_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def tune_five(X, Y, seed):
    """run_q1.tune logic, with five instead of three inner folds."""
    inner = list(KFold(5, shuffle=True, random_state=seed).split(X))
    candidates, rows = {}, []
    for ix, spec in enumerate(specifications()):
        pp = np.empty_like(Y)
        for tr, va in inner:
            pp[va] = SensoryEstimator(spec).fit(X[tr], Y[tr]).predict(X[va])
        score = float(np.mean(((pp-Y)/np.std(Y, axis=0, ddof=1))**2))
        f = spec['family']
        rows.append({'spec': spec, 'inner_nmse': score})
        if f not in candidates or score < candidates[f][0]:
            candidates[f] = (score, spec, pp)
        print(f'candidate {ix+1}/22 {spec} NMSE={score:.6f}', flush=True)
    weights = learn_weights([candidates[f][2] for f in FAMILIES], Y)
    return candidates, weights, rows, inner


def main(args):
    started = time.time()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    x = pd.read_csv(data/'1.csv')  # Intentionally no sort_values.
    y = pd.read_csv(data/'4.csv').set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.beer.to_numpy(), y.beer.to_numpy())
    features, targets = x.columns[3:].tolist(), y.columns[3:].tolist()
    X, Y = x[features].to_numpy(float), y[targets].to_numpy(float)
    tr, te = train_test_split(np.arange(len(x)), test_size=.30, random_state=0,
                             shuffle=True, stratify=x.tasting_category_fine)
    split = x[['beer', 'tasting_category_fine']].copy()
    split['source_row'] = np.arange(len(x)); split['partition'] = 'train'
    split.loc[te, 'partition'] = 'test'
    split.to_csv(out/'author_split_assignments.csv', index=False)
    cand, weights, rows, inner = tune_five(X[tr], Y[tr], 20261023)
    models = [SensoryEstimator(cand[f][1]).fit(X[tr], Y[tr]) for f in FAMILIES]
    predictions = {f: m.predict(X[te]) for f, m in zip(FAMILIES, models)}
    predictions['ensemble'] = sum(weights[i]*predictions[f] for i,f in enumerate(FAMILIES))
    predictions['training_mean'] = np.broadcast_to(Y[tr].mean(axis=0), Y[te].shape).copy()
    predictions['style_mean'] = np.array([
        Y[tr][x.iloc[tr].tasting_category_fine.to_numpy() == x.iloc[i].tasting_category_fine].mean(axis=0)
        for i in te])
    scores = {f: strict_jsonable(jsonable(assess(Y[te], p))) for f,p in predictions.items()}
    scalar_scores = [{k:v for k,v in dict(model=f, **s).items() if k not in ['r2','rmse']}
                     for f,s in scores.items()]
    pd.DataFrame(scalar_scores).to_csv(out/'holdout_model_comparison.csv', index=False)
    metric_rows = [{'model': f, 'target': t, 'r2': s['r2'][j], 'rmse': s['rmse'][j]}
                   for f,s in scores.items() for j,t in enumerate(targets)]
    pd.DataFrame(metric_rows).to_csv(out/'holdout_dimension_metrics.csv', index=False)
    for f,p in predictions.items():
        pd.concat([x.iloc[te][['beer','tasting_category_fine']].reset_index(drop=True),
                   pd.DataFrame(p,columns=targets)],axis=1).to_csv(out/f'holdout_predictions_{f}.csv',index=False)
    inner_predictions = {f:cand[f][2] for f in FAMILIES}
    inner_predictions['ensemble'] = sum(w*inner_predictions[f] for w,f in zip(weights,FAMILIES))
    inner_id = np.full(len(tr), -1)
    for k,(_,va) in enumerate(inner): inner_id[va] = k
    inner_meta = x.iloc[tr][['beer','tasting_category_fine']].reset_index(drop=True)
    inner_meta['inner_fold'] = inner_id
    for f,p in inner_predictions.items():
        pd.concat([inner_meta,pd.DataFrame(p,columns=targets)],axis=1).to_csv(
            out/f'training_inner_oof_selected_{f}.csv',index=False)
    np.savez_compressed(out/'holdout_arrays.npz',Y=Y[te],beer=x.iloc[te].beer.to_numpy(),
                        train_rows=tr,test_rows=te,**predictions)
    np.savez_compressed(out/'inner_selection_arrays.npz',Y=Y[tr],beer=x.iloc[tr].beer.to_numpy(),
                        inner_fold=inner_id,**inner_predictions)
    joblib.dump({'features':features,'targets':targets,'families':FAMILIES,'models':models,
                 'weights':weights,'reference_y':Y[tr],'source':'strict author-split comparison'},
                out/'author_split_existing_ensemble.joblib',compress=3)
    workbook = ROOT/'audit/external/original_supplement.xlsx'
    published = pd.read_excel(workbook,sheet_name='Supplementary Table S3')
    names = pd.read_excel(workbook,sheet_name='Sensory names').query("dataset == 'Trained panel'").set_index('R_name')
    expected = names.loc[targets,'CleanName_PanelFeedback'].tolist()
    assert expected == published.aspect.tolist(), 'Published S3 target order does not match'
    comp = published.rename(columns={c:'paper_individual_'+c for c in published.columns if c!='aspect'})
    comp.insert(0,'target',targets)
    for f,s in scores.items():comp['current_'+f] = s['r2']
    for f in ['GradientBoost','ExtraTrees','RandomForest']:
        comp['ensemble_delta_vs_paper_individual_'+f] = comp.current_ensemble-comp['paper_individual_'+f]
    comp.to_csv(out/'published_s3_target_comparison.csv',index=False)
    selected = {'specs':{f:cand[f][1] for f in FAMILIES},'weights':dict(zip(FAMILIES,weights.tolist())),
                'tuning':rows}
    (out/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    summary = {
        'interpretation':'Same published 175/75 stratified split with our existing pipeline and strict fold-local preprocessing; not a reproduction of all original model searches.',
        'split':{'test_size':.30,'random_state':0,'stratify':'tasting_category_fine','source_row_order_retained':True,'train_n':len(tr),'test_n':len(te)},
        'inner_cv':{'n_splits':5,'shuffle':True,'random_state':20261023,'selection':'pooled inner prediction standardized MSE, unchanged from existing pipeline'},
        'paper_inner_cv_difference':'Paper GridSearchCV(cv=5) defaults to unshuffled KFold and mean fold R2; this run deliberately preserves our own selection loss and shuffle.',
        'n_features':len(features),'n_targets':len(targets),'holdout_scores':scores,
        'selected':{k:v for k,v in selected.items() if k!='tuning'},
        'paper_individual_comparison':{
            f:{'published_mean_rounded_r2':float(published[f].mean()),
               'ensemble_mean_r2_difference':float(comp['ensemble_delta_vs_paper_individual_'+f].mean()),
               'ensemble_larger_count':int((comp['ensemble_delta_vs_paper_individual_'+f]>0).sum())}
            for f in ['GradientBoost','ExtraTrees','RandomForest']},
        'comparison_caution':'Published S3 is rounded per-target independently selected models; Table 1 uses different multi-output models. All 250 labels were available during prior development, so this retrospective split is not a new blinded test.',
        'wall_seconds':time.time()-started}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'scores':scalar_scores,'weights':selected['weights'],'elapsed':summary['wall_seconds']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(resolve_data_dir()),help=DATA_HELP)
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'author_protocol_baseline'))
    with threadpool_limits(limits=1):main(p.parse_args())
