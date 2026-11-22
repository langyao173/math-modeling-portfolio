"""Run licensed Kaiwu SA locally on the six application matrices.

Dry-run is the default. No cloud optimizer is used and no quantum credits are
spent. Credentials remain in the local SDK license/environment.
"""
from pathlib import Path
import argparse,csv,subprocess,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.qubo import read_matrix,preflight_ising
from qpool.results import read_solutions,inspect_solutions

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
 p.add_argument('--matrix-dir',type=Path,default=Path('upload/03_application_batch'))
 p.add_argument('--out',type=Path,default=Path('results/kaiwu_application_sa'))
 p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);a=p.parse_args()
 cases=list(csv.DictReader((a.matrix_dir/'manifest.csv').open()))
 for case in cases:
  mat=a.matrix_dir/(case['name']+'_ising.csv');pf=preflight_ising(read_matrix(mat))
  if not pf['valid']:raise SystemExit(str(mat)+': '+str(pf['issues']))
 print(f'{len(cases)} checked matrices; {len(cases)*len(a.seeds)} planned local SA runs; zero cloud submissions.')
 if not a.execute:
  print('Add --execute in your licensed Python 3.10 environment to run official Kaiwu SA.');return
 if sys.version_info[:2]!=(3,10):raise SystemExit('Use Python 3.10 with the officially licensed Kaiwu installation.')
 runner=Path(__file__).with_name('kaiwu_bridge.py')
 summaries=[]
 for c in cases:
  name=c['name'];matrix=a.matrix_dir/(name+'_ising.csv');meta=a.matrix_dir/(name+'.json')
  for seed in a.seeds:
   out=a.out/f'{name}_s{seed}';out.mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,str(runner),'--matrix',str(matrix),'--action','sa','--task-name',name,'--seed',str(seed),'--out',str(out)]
   subprocess.run(cmd,check=True)
   report=inspect_solutions(meta,read_solutions(out/'kaiwu_sa_spins.csv'),source='official_Kaiwu_SA_local_classical')
   (out/'checked_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
   run=json.loads((out/'kaiwu_sa_run.json').read_text())
   summaries.append({'name':name,'seed':seed,'elapsed_s':run['elapsed_s'],'sdk_version':run['sdk_version'],
                     'best_raw_reward':report['best_raw_feasible_reward'],'raw_feasible_rate':report['raw_feasible_rate'],
                     'n_returned_rows':report['n_returned_rows'],'is_quantum':False})
 a.out.mkdir(parents=True,exist_ok=True)
 with (a.out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 print('Finished local Kaiwu SA. Preserve the entire results folder, including raw spins and settings.')
if __name__=='__main__':main()
