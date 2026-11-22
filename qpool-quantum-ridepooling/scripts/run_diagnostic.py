from pathlib import Path
import sys,argparse,csv,json,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.replay import Config,run_replay
import scipy,numpy

def main():
    p=argparse.ArgumentParser(description='Independent replay on an externally supplied example; not a native-SOTA benchmark.')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/replay'))
    p.add_argument('--fleets',default='4,8,12');p.add_argument('--seeds',default='11,22,33')
    p.add_argument('--methods',default='greedy,local_search,myopic_mip,point_mip,separable_tv_mip,joint_tv_mip')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    env={'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,
         'kaiwu_executed':False,'quantum_executed':False,'thread_guidance':'Set OPENBLAS_NUM_THREADS=1; actual configured environment recorded below.'}
    import os
    env['OPENBLAS_NUM_THREADS']=os.environ.get('OPENBLAS_NUM_THREADS')
    (a.output/'environment.json').write_text(json.dumps(env,indent=2))
    for fleet in map(int,a.fleets.split(',')):
        for seed in map(int,a.seeds.split(',')):
            for method in a.methods.split(','):
                cfg=Config(fleet=fleet,seed=seed)
                name=f'f{fleet}_s{seed}_{method}'
                r=run_replay(a.data,a.output/name,method,cfg,capture=method=='myopic_mip')
                row={k:v for k,v in r.items() if k!='config'};row.update(fleet=fleet,seed=seed)
                rows.append(row)
                with (a.output/'summary.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                print(name,r['completed'],r['max_candidates'],round(r['wall_clock_s'],3),flush=True)
if __name__=='__main__':main()
