from pathlib import Path
import sys,json,csv,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.replay import Config,run_replay
root=Path('results/stage2/lns_replay');root.mkdir(parents=True,exist_ok=True)
(root/'design.json').write_text(json.dumps({'study':'second development iteration, not sealed test','method':'qpool_lns_cpu','fleets':[8,16,32,48],'epochs':[60,300],'seeds':[11,22,33],'candidate_pool_unchanged':True,'quantum_calls':0,'minimum_precision':1,'max_core_variables':96,'safeguard':'hold local-search seed on objective ties'},indent=2))
rows=[]
for f,e,s in itertools.product([8,16,32,48],[60,300],[11,22,33]):
 name=f'f{f}_e{e}_s{s}_qpool_lns_cpu'
 m=run_replay('data/private_example',root/name,'qpool_lns_cpu',Config(fleet=f,epoch=e,seed=s))
 rows.append({k:v for k,v in m.items() if k!='config'}|{'fleet':f,'epoch':e,'seed':s,'run':name})
 print(name,m['completed'],m['vehicle_driving_s'],flush=True)
with (root/'all_runs.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('TOTAL',sum(r['completed'] for r in rows),sum(r['vehicle_driving_s'] for r in rows))
