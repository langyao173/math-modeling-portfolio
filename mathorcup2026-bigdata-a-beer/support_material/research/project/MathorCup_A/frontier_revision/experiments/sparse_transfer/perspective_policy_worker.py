"""One policy per worker: no overlap in case output directories."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from control_problem import FixedProblem
from perspective_solver import run_case,OUT,write_json

parser=argparse.ArgumentParser()
parser.add_argument('--policy',choices=['strict','exceedance'],required=True)
parser.add_argument('--seconds',type=float,default=180.)
args=parser.parse_args()
with threadpool_limits(limits=1):
    problem=FixedProblem(args.policy,style_epsilon=1e-5)
    results=[]
    for beer in problem.contexts:
        results.append(run_case(args.policy,beer,args.seconds))
        pd.DataFrame(results).to_csv(OUT/f'comparison_{args.policy}.csv',index=False)
    write_json(OUT/f'DONE_{args.policy}.json',dict(policy=args.policy,cases=len(results),seconds_per_case=args.seconds))
print('POLICY_COMPLETE '+json.dumps(dict(policy=args.policy,cases=len(results))),flush=True)
