"""Check original-utility loss on the exact quantized-optimal face (CPU only)."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import vstack,csr_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qpool.solvers import packing_constraints
from qpool.qubo import PackingInstance
rows=[]
for p in sorted((ROOT/'upload/03_application_batch').glob('QP_A??.json')):
    m=json.loads(p.read_text());q=PackingInstance.from_dict(m)
    A=packing_constraints(q);n=len(q.weights)
    constraint=vstack([A,csr_matrix(q.weights.reshape(1,-1))],format='csc')
    lower=np.r_[np.full(A.shape[0],-np.inf),m['exact_quantized_optimum']]
    upper=np.r_[np.ones(A.shape[0]),m['exact_quantized_optimum']]
    solved=milp(-np.asarray(m['original_weights']),integrality=np.ones(n),bounds=Bounds(0,1),
        constraints=LinearConstraint(constraint,lower,upper),options={'time_limit':20.,'mip_rel_gap':0.})
    if solved.status!=0:raise RuntimeError('Exact reference not established for '+m['name'])
    x=np.rint(solved.x).astype(int)
    utility=float(np.asarray(m['original_weights'])@x)+m['fixed_original_utility']
    row={'name':m['name'],'best_original_utility_on_quantized_optimal_face':utility,
         'conditional_original_optimum':m['conditional_original_reference_utility'],
         'unavoidable_loss_if_quantized_optimality_retained':m['conditional_original_reference_utility']-utility,
         'proven_optimal':True,'quantized_reward':float(q.weights@x),'bits':x.tolist()}
    rows.append(row)
    print(m['name'],utility,row['unavoidable_loss_if_quantized_optimality_retained'])
( ROOT/'results/stage3/verified/optimal_face_recheck.json').write_text(json.dumps(rows,indent=2))
