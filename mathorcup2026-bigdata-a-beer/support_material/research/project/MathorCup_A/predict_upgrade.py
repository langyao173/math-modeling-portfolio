#!/usr/bin/env python3
"""Predict 50 standardized sensory targets using the eight-base research model.

These values are not reconstructed original 0-6 taster scores.
"""
from pathlib import Path
import argparse
import os
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import pandas as pd

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT/'A_work/nature_upgrade/fusion')
    p.add_argument('--rule',choices=['target_shrink','global'],default='target_shrink')
    args=p.parse_args()
    folder=args.model_dir.resolve()
    sys.path.insert(0,str(folder))
    from fusion_model import load_fusion
    model=load_fusion(folder)
    frame=pd.read_csv(args.input)
    missing=[f for f in model.bundle['features'] if f not in frame]
    if missing:p.error('Missing required columns: '+', '.join(missing))
    pred=pd.DataFrame(model.predict(frame,rule=args.rule),columns=model.bundle['targets'])
    ids=[c for c in ['beer','beer_id','tasting_category_fine'] if c in frame]
    result=pd.concat([frame[ids].reset_index(drop=True),pred],axis=1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Wrote {len(result)} rows and 50 sensory targets to {args.output}')


if __name__=='__main__':main()
