"""Matched exploratory competition replay, not a sealed native leaderboard."""
from pathlib import Path
import sys,json,csv,itertools,time,platform,argparse
import numpy as np,scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.replay import run_replay,Config

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/private_example'))
 p.add_argument('--out',type=Path,default=Path('results/stage2/contest_replay'));a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 methods=['greedy','local_search','myopic_mip','qpool_cpu'];fleets=[8,16,32,48];epochs=[60,300];seeds=[11,22,33]
 design={'methods':methods,'fleets':fleets,'epochs_seconds':epochs,'seeds_initial_vehicle_positions':seeds,
  'common_config':{'capacity':4,'max_wait':600,'max_detour':600,'dwell':60,'cost_weight':.08,'max_new_requests':3,'solver_budget':1.},
  'trace':'external Sioux Falls example, first 240 requests; original arrival times unchanged',
  'study_type':'competition development; not a sealed test or native benchmark reproduction',
  'quantum_backend_used_in_these_runs':False,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
 (a.out/'design.json').write_text(json.dumps(design,indent=2));rows=[]
 for f,e,s,method in itertools.product(fleets,epochs,seeds,methods):
  name=f'f{f}_e{e}_s{s}_{method}';out=a.out/name
  cfg=Config(fleet=f,epoch=e,seed=s,solver_budget=1.,cost_weight=.08)
  m=run_replay(a.data,out,method,cfg,capture=(method=='myopic_mip' and s==11))
  rows.append({k:v for k,v in m.items() if k!='config'}|{'fleet':f,'epoch':e,'seed':s,'run':name})
  print(name,m['completed'],round(m['wall_clock_s'],3),flush=True)
 with (a.out/'all_runs.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for method in methods:
  rs=[r for r in rows if r['method']==method]
  summary.append({'method':method,'runs':len(rs),'completed_sum':sum(r['completed'] for r in rs),'requests_sum':sum(r['n_requests'] for r in rs),
   'service_rate_pooled':sum(r['completed'] for r in rs)/sum(r['n_requests'] for r in rs),
   'vehicle_driving_s_sum':sum(r['vehicle_driving_s'] for r in rs),'empty_driving_s_sum':sum(r['empty_driving_s'] for r in rs),
   'decision_p95_s_median_across_runs':float(np.median([r['decision_p95_s'] for r in rs])),
   'all_completed_or_rejected':all(r['unfinished']==0 for r in rs)})
 with (a.out/'summary.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
