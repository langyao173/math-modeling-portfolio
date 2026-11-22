"""Predict 50 sensory responses and explicitly relative 0--6 grades.

Input columns and scales must follow attachment 1. This script does not infer
whether a user's concentrations have already been log-transformed.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sensory_model import predict_bundle

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--imputer',choices=['median','knn'],default='median')
    ap.add_argument('--model',type=Path)
    args=ap.parse_args()
    model=args.model or Path(__file__).with_name('sensory_ensemble_knn.joblib' if args.imputer=='knn' else 'sensory_ensemble.joblib')
    bundle=joblib.load(model)
    source=pd.read_csv(args.input)
    if not len(source):
        raise ValueError('Input table has no rows.')
    features=bundle['features']
    available=[c for c in features if c in source]
    if not available:
        raise ValueError('No attachment-1 chemical feature names are present.')
    frame=source.reindex(columns=features).apply(pd.to_numeric,errors='raise')
    frame=frame.replace([np.inf,-np.inf],np.nan)
    if frame.isna().all(axis=1).any():
        raise ValueError('At least one input row has no observed chemical measurements.')
    scores,grades=predict_bundle(bundle,frame,include_grades=True)
    meta=source[[c for c in ['beer','beer_id','tasting_category_fine','synthetic_id'] if c in source]].copy()
    meta.insert(0,'input_row',np.arange(1,len(source)+1))
    meta['missing_fraction']=frame.isna().mean(axis=1)
    meta['severe_missingness_flag']=meta.missing_fraction>.5
    args.output.mkdir(parents=True,exist_ok=True)
    pd.concat([meta,scores],axis=1).to_csv(args.output/'sensory_standardized.csv',index=False)
    pd.concat([meta,grades],axis=1).to_csv(args.output/'sensory_relative_grades_0_6.csv',index=False)
    report=dict(n_rows=len(source),n_targets=len(bundle['targets']),imputer=args.imputer,
        missing_columns=[c for c in features if c not in source],
        grade_interpretation='Relative within-study intensity grades, not recovered original panel scores.',
        prediction_interpretation='Predictions from the full-data fitted model, not out-of-fold performance estimates.',
        input_scale='Attachment 1 column scales; no automatic uniform log or inverse-log transformation.',
        model_file=str(model.name))
    (args.output/'prediction_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    main()
