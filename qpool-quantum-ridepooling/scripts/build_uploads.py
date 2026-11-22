from pathlib import Path
import sys,csv,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.qubo import *
from qpool.solvers import solve_milp,solve_local,solve_sa


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,default=Path('results/replay'))
    p.add_argument('--out',type=Path,default=Path('upload'));p.add_argument('--results',type=Path,default=Path('results/kernels'))
    a=p.parse_args();a.results.mkdir(parents=True,exist_ok=True)
    smoke=PackingInstance([8,7,6,5,4,3],[(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)])
    opt,winners=exact_enumeration(smoke)
    m=export_instance(smoke,a.out/'01_calibration','QP_CHECK06',{'purpose':'calibration_only_not_application_benchmark',
        'exact_optimal_reward':opt,'exact_optimal_bits':winners.tolist()})
    print('Calibration',opt,winners.tolist())
    pool=[]
    # Predefined chronological snapshots from a single development initialization.
    for fleet in (4,8,12):
        f=a.replay/f'f{fleet}_s11_myopic_mip'/'snapshots.json'
        snaps=json.loads(f.read_text())
        for d in [s for s in snaps if len(s['weights'])>=8][:4]:pool.append((fleet,d))
    rows=[];chosen=[];bench=[]
    for index,(fleet,snap) in enumerate(pool):
        inst=PackingInstance.from_dict(snap);xref,info=solve_milp(inst,10);original_opt=float(inst.weights@xref)
        raw_ok=False
        try:raw_ok=compile_instance(inst)[2]['preflight']['valid']
        except ValueError:pass
        viable=[]
        for resolution in (1,2,4,8,16,32):
            quant=PackingInstance(np.rint(inst.weights*resolution),inst.edges,inst.labels)
            Q,I,meta=compile_instance(quant);pf=meta['preflight']
            x,qinfo=solve_milp(quant,10);actual=float(inst.weights@x)
            K=min(fleet,len(inst.weights))
            row={'kernel_index':index,'fleet':fleet,'epoch':snap['epoch'],'n_variables':len(inst.weights),
                 'n_edges':len(inst.edges),'resolution':resolution,'preflight_pass':pf['valid'],
                 'minimum_ising':pf['minimum'],'maximum_ising':pf['maximum'],'original_exact_precision_pass':raw_ok,
                 'original_optimal_utility':original_opt,'quantized_choice_original_utility':actual,
                 'original_utility_regret':original_opt-actual,'worst_case_regret_bound':K/resolution,
                 'quantized_solve_optimal':qinfo['optimal']}
            assert original_opt-actual<=K/resolution+1e-7
            rows.append(row)
            if pf['valid']:viable.append((resolution,quant,row))
        if not viable:continue
        resolution,quant,row=max(viable,key=lambda t:t[0]) # precision only; no selection by reward advantage
        for name,solver in [('HiGHS',lambda:solve_milp(quant,10)),('local_search',lambda:solve_local(quant,.2))]:
            x,inf=solver();bench.append({'kernel_index':index,'solver':name,'seed':-1,'reward':float(quant.weights@x),
                'feasible':quant.evaluate(x)['feasible'],'elapsed_s':inf['elapsed_s'],'optimal':inf['optimal']})
        for seed in (1,2,3):
            x,inf=solve_sa(quant,seed,100,4);bench.append({'kernel_index':index,'solver':'local_Metropolis_SA_NOT_Kaiwu','seed':seed,
                'reward':float(quant.weights@x),'feasible':quant.evaluate(x)['feasible'],'elapsed_s':inf['elapsed_s'],'optimal':False})
        # At most two application kernels in the initial upload folder; do not burn credits on easy cases.
        if len(chosen)<2:
            name=f'QP_SF_F{fleet}_T{snap["epoch"]}'
            m=export_instance(quant,a.out/'02_example_kernels',name,{
                'purpose':'external_example_conditional_kernel_not_native_benchmark',
                'source_replay':f'f{fleet}_s11_myopic_mip','epoch':snap['epoch'],
                'original_weights':inst.weights.tolist(),'utility_quantization_resolution':resolution,
                'original_optimal_utility':original_opt,'quantization_worst_case_regret_bound':fleet/resolution,
                'full_day_quantum_replay':False,'exact_quantized_optimum':float(quant.weights@solve_milp(quant,10)[0])})
            chosen.append({'name':name,'n_variables':len(inst.weights),'resolution':resolution,'meta':m})
    for name,data in [('precision_sweep',rows),('classical_kernel_runs',bench)]:
        if data:
            with (a.results/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (a.results/'initial_uploads.json').write_text(json.dumps(chosen,indent=2))
    print('Snapshots',len(pool),'precision experiments',len(rows),'classical kernel runs',len(bench))
    print('Application uploads',[(d['name'],d['n_variables']) for d in chosen])
if __name__=='__main__':main()
