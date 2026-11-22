"""Predict 50 continuous attributes and train-referenced relative grades."""
from pathlib import Path
import argparse
import pandas as pd
from fit_final import predict_semantic_bundle

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('chemistry_csv',type=Path)
    a.add_argument('--model',type=Path,default=Path(__file__).with_name('semantic_corrected_predictor.joblib'))
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();df=pd.read_csv(args.chemistry_csv)
    continuous,grades=predict_semantic_bundle(args.model,df,True)
    if 'beer' in df:
        continuous.insert(0,'beer',df.beer.to_numpy());grades.insert(0,'beer',df.beer.to_numpy())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    continuous.to_csv(args.output,index=False)
    grades.to_csv(args.output.with_name(args.output.stem+'_relative_grades.csv'),index=False)
