"""Fit a KNN-imputation inference variant with unchanged final specs/weights.
This is an imputation sensitivity variant, not another independently selected
or validated winner. Its strict missingness results are stored in audit/.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,joblib
import pandas as pd
from sensory_model import SensoryEstimator

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(os.environ.get('MATHORCUP_DATA_DIR','/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件')))
    ap.add_argument('--output',type=Path,default=Path(__file__).parent)
    args=ap.parse_args()
    base=joblib.load(args.output/'sensory_ensemble.joblib')
    x=pd.read_csv(args.data/'1.csv').sort_values('beer')
    y=pd.read_csv(args.data/'4.csv').set_index('beer').loc[x.beer]
    models=[SensoryEstimator(s,imputer='knn').fit(x[base['features']].to_numpy(float),y[base['targets']].to_numpy(float)) for s in base['specs']]
    new=dict(base,models=models,imputation='KNN n_neighbors=7 weights=distance; same specifications and ensemble weights')
    joblib.dump(new,args.output/'sensory_ensemble_knn.joblib')
    print('Wrote sensory_ensemble_knn.joblib')

if __name__=='__main__':
    main()
