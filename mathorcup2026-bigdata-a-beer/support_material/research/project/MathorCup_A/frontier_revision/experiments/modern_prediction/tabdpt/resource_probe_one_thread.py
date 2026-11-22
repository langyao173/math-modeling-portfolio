"""Separate one-thread timing only; does not create/change the frozen benchmark."""
import os
os.environ['BEER_TABDPT_THREADS']='1'
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_tabdpt_benchmark import load_data,predict_one,HERE

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with threadpool_limits(limits=1):
    x,y,X,Y,fold=load_data()
    tr=np.flatnonzero(fold!=0);te=np.flatnonzero(fold==0)
    prediction,projection,timing=predict_one(X,Y,tr,te,0,0)
record=dict(**timing,heldout_scores_computed=False,
    scope='Independent one-thread resource probe; the formal benchmark remains six threads and repeats its own pilot.',
    linear_250_target_minutes=timing['total_s']*250/60,
    estimate_limit='One-thread linear projection is a planning reference, not a measured six-thread benchmark runtime.')
np.savez_compressed(HERE/'resource_probe_one_thread.npz',prediction=prediction,projection=projection,
                    train_indices=tr,test_indices=te)
(HERE/'resource_probe_one_thread.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
