"""Equal-budget development tuning; fresh initialization-seed checks, NOT new data days."""
from pathlib import Path
import sys,csv,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.replay import Config,run_replay

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('results/price_race'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    design={'methods':['myopic_mip','adaptive_price_mip'],'coefficients':[0,.08,.2,.4,.8,1.2],
            'development_seeds':[11,22],'evaluation_seeds':[44,55,66],'fleets':[4,8,12],
            'selection':'maximum total completed; ties smaller coefficient','same_source_trace':True,
            'interpretation':'only generalization to fleet initialization; not independent days or native baseline tuning'}
    (a.out/'design.json').write_text(json.dumps(design,indent=2));rows=[];selected={}
    for method in design['methods']:
        scores=[]
        for k in design['coefficients']:
            completed=0
            for fleet in design['fleets']:
                for seed in design['development_seeds']:
                    cfg=Config(fleet=fleet,seed=seed,cost_weight=k)
                    folder=a.out/f'dev_{method}_k{k}_f{fleet}_s{seed}'
                    r=run_replay(a.data,folder,method,cfg)
                    row={key:v for key,v in r.items() if key!='config'};row.update(split='development',coefficient=k,fleet=fleet,seed=seed)
                    rows.append(row);completed+=r['completed']
            scores.append((completed,-k));print(method,k,completed,flush=True)
        selected[method]=-max(scores)[1]
    (a.out/'selected.json').write_text(json.dumps(selected,indent=2))
    # No coefficient changes after reading these seed results.
    for method,k in selected.items():
        for fleet in design['fleets']:
            for seed in design['evaluation_seeds']:
                r=run_replay(a.data,a.out/f'eval_{method}_f{fleet}_s{seed}',method,Config(fleet=fleet,seed=seed,cost_weight=k))
                row={key:v for key,v in r.items() if key!='config'};row.update(split='evaluation',coefficient=k,fleet=fleet,seed=seed);rows.append(row)
                print('EVAL',method,fleet,seed,r['completed'],flush=True)
    with (a.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
